import numba
import torch
import numpy as np
from torch import nn
import torchvision
from math import sqrt as sqrt
from utils import *
import time
from scipy import stats as stats
from torch.nn import functional as F

""" 使用numba将python函数编译为机器代码加速计算 """


@numba.jit(nopython=True)
def make_input_tensor(input_tensor, new_aug_lidar_cam_coords, bin_idxs, pillar_idxs):
    """
    :return input_tensor: (10, P, N) np.array passed into conv layer.
    """
    max_pts_per_pillar = 100
    num_nonempty_pillars = pillar_idxs.shape[0]
    for i in range(num_nonempty_pillars):
        condition = bin_idxs == pillar_idxs[i]
        condition = (condition[:, 0] & condition[:, 1])
        # all points w/ same bin idx as pillar_idxs[i]
        points = new_aug_lidar_cam_coords[condition][:max_pts_per_pillar]
        points = points.T
        num_points = points.shape[1]
        input_tensor[:, i, :num_points] = points

    print("model.py: Break Point9")
    return input_tensor


""" pillar编码特征层 """


class PFNv2(nn.Module):
    def __init__(self):
        super(PFNv2, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_pillars = 12000
        self.max_pts_per_pillar = 100
        self.xrange = (-40, 40)
        self.zrange = (0, 80)

        # output is (batch, 64, P, N) tensor
        self.conv1 = nn.Conv2d(10, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, lidar):
        """
        :param lidar: list of tensors. (N_points, 5) in camera coordinates (x,y,z,r,class)
        """
        # print("S6 =", lidar)
        assert isinstance(lidar, list)

        lidar_one = np.array(lidar[0])
        # print("S7 =", lidar_one)
        assert len(lidar_one.shape) == 2

        batch_size = len(lidar)  # lidar batch size is 4
        input_tensor = torch.zeros(batch_size, 10, self.max_pillars, self.max_pts_per_pillar).to(self.device)
        encoded_bev = torch.zeros(batch_size, 64, 500, 500).to(self.device)
        pillar_idxs = []

        """ 遍历批次内的雷达数据，对每一个批次的数据进行pillar操作 """
        for i in range(batch_size):
            # print("S8 =", lidar[i])

            l = np.array(lidar[i])
            # print("S9 =", l)

            """ pillar设置 """
            new_aug_lidar_cam_coords, bin_idxs, pillar_idxs_ = self.augment_points(l)
            pillar_idxs.append(pillar_idxs_)
            input_tensor_ = np.zeros((10, self.max_pillars, self.max_pts_per_pillar))
            input_tensor_ = make_input_tensor(input_tensor_, new_aug_lidar_cam_coords, bin_idxs, pillar_idxs_)
            input_tensor[i] = torch.from_numpy(input_tensor_).to(self.device)

        x = F.relu(self.bn1(self.conv1(input_tensor)))  # (batch, 64, P, N)
        encoded_pillars, _ = x.max(dim=-1)  # (batch, 64, P)

        for i in range(batch_size):
            encoded_bev_ = self.scatter_encoded_pillars_to_bev(encoded_pillars[i], pillar_idxs[i])
            encoded_bev[i] = encoded_bev_

        print("model.py: Break Point1")
        return encoded_bev  # (batch,p64, 500, 500)

    def augment_points(self, augmented_lidar_cam_coords):
        """
        Converts (x,y,z,r,class) to (x,y,z,r,class,xc,yc,zc,xp,zp)
        """

        # print("S10 =", augmented_lidar_cam_coords)

        points_in_xrange = (-40 < augmented_lidar_cam_coords[:, 0]) & (augmented_lidar_cam_coords[:, 0] < 40)
        points_in_zrange = (0 < augmented_lidar_cam_coords[:, 2]) & (augmented_lidar_cam_coords[:, 2] < 70.4)
        augmented_lidar_cam_coords = augmented_lidar_cam_coords[points_in_xrange & points_in_zrange]
        new_aug_lidar_cam_coords = np.zeros((augmented_lidar_cam_coords.shape[0], 10))
        new_aug_lidar_cam_coords[:, :5] = augmented_lidar_cam_coords

        xedges = np.linspace(-40, 40, 501, dtype=np.float32)
        # 80 first because a point 80m from ego car is in top row of bev img (row 0)
        zedges = np.linspace(80, 0, 501, dtype=np.float32)
        x = augmented_lidar_cam_coords[:, 0]  # left/right
        # y in cam coords (+ is down, - is up)
        y = augmented_lidar_cam_coords[:, 1]
        z = augmented_lidar_cam_coords[:, 2]  # front/back

        x_inds = np.digitize(x, xedges).reshape(-1, 1) - \
                 1  # subtract 1 to get 0 based indexing
        z_inds = np.digitize(z, zedges).reshape(-1, 1) - \
                 1  # idx into rows of bev img
        # z first because it corresponds to rows of the bev
        bin_idxs = np.hstack((z_inds, x_inds))

        ret_x = stats.binned_statistic_2d(z, x, x, 'mean', bins=[np.flip(zedges), xedges])  # mean of x vals of points in each bin
        ret_y = stats.binned_statistic_2d(
            z, x, y, 'mean', bins=[np.flip(zedges), xedges])
        ret_z = stats.binned_statistic_2d(
            z, x, z, 'mean', bins=[np.flip(zedges), xedges])
        # since need to flip zedges (row bins) to make function work, need to flip output by rows
        x_mean = np.flip(ret_x.statistic, axis=0)
        y_mean = np.flip(ret_y.statistic, axis=0)
        # mean of all z values in each bev img 'pixel', NaN at cells with no points.
        z_mean = np.flip(ret_z.statistic, axis=0)

        # coord of x center of each bev cell. All cols have same value
        x_ctr = np.tile(np.linspace((-40 + .08), (40 - .08), 500), (500, 1))
        # all rows have same value
        z_ctr = np.tile(np.linspace((80 - .08), (0 + .08),
                                    500).reshape(-1, 1), (1, 500))

        # offset of each point from x_mean of pillar, ymean, zmean, x_center, y_center
        # offset of each point from xmean of pillar
        new_aug_lidar_cam_coords[:, 5] = new_aug_lidar_cam_coords[:,
                                         0] - x_mean[bin_idxs[:, 0], bin_idxs[:, 1]]
        new_aug_lidar_cam_coords[:, 6] = new_aug_lidar_cam_coords[:,
                                         1] - y_mean[bin_idxs[:, 0], bin_idxs[:, 1]]  # yc
        new_aug_lidar_cam_coords[:, 7] = new_aug_lidar_cam_coords[:,
                                         2] - z_mean[bin_idxs[:, 0], bin_idxs[:, 1]]  # zc
        new_aug_lidar_cam_coords[:, 8] = new_aug_lidar_cam_coords[:, 0] - \
                                         x_ctr[bin_idxs[:, 0], bin_idxs[:, 1]
                                         ]  # offset from x center of pillar
        new_aug_lidar_cam_coords[:, 9] = new_aug_lidar_cam_coords[:,
                                         2] - z_ctr[bin_idxs[:, 0], bin_idxs[:, 1]]  # zp

        H, _, __ = np.histogram2d(z, x, bins=(np.flip(zedges), xedges))
        H[H != 0] = 1
        # pillars containing >= 1 lidar point
        num_nonempty_pillars = int(np.flip(H, axis=0).sum())

        # ith element will be bin (row, col of bev img) of that pillar
        pillar_idxs = np.unique(bin_idxs, axis=0)
        if pillar_idxs.shape[0] > self.max_pillars:
            np.random.shuffle(pillar_idxs)
            pillar_idxs = pillar_idxs[:self.max_pillars]

        print("model.py: Break Point2")

        return new_aug_lidar_cam_coords, bin_idxs, pillar_idxs

    def scatter_encoded_pillars_to_bev(self, encoded_pillars, pillar_idxs):
        """
        :return encoded_bev: (64, 500, 500) tensor for input to resnet portion of network
        """
        num_nonempty_pillars = pillar_idxs.shape[0]
        # bev_map and encoded_pillars must be torch.float, indices tensors must be torch.long
        encoded_bev = torch.zeros(64, 500, 500).to(self.device)
        encoded_bev[:, pillar_idxs[:, 0], pillar_idxs[:, 1]
        ] = encoded_pillars[:, :num_nonempty_pillars]

        print("model.py: Break Point3")

        return encoded_bev


class PredictionConvolutions(nn.Module):
    def __init__(self, channels_for_block, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes  # including background

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = 2

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes (4 values) and height/elevation (2 values))
        self.loc_block0 = nn.Conv2d(
            channels_for_block[0], n_boxes * 4, kernel_size=3, padding=1)
        self.loc_block1 = nn.Conv2d(
            channels_for_block[1], n_boxes * 4, kernel_size=3, padding=1)
        self.loc_block2 = nn.Conv2d(
            channels_for_block[2], n_boxes * 4, kernel_size=3, padding=1)
        self.loc_block3 = nn.Conv2d(
            channels_for_block[3], n_boxes * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_block0 = nn.Conv2d(
            channels_for_block[0], n_boxes * n_classes, kernel_size=3, padding=1)
        self.cl_block1 = nn.Conv2d(
            channels_for_block[1], n_boxes * n_classes, kernel_size=3, padding=1)
        self.cl_block2 = nn.Conv2d(
            channels_for_block[2], n_boxes * n_classes, kernel_size=3, padding=1)
        self.cl_block3 = nn.Conv2d(
            channels_for_block[3], n_boxes * n_classes, kernel_size=3, padding=1)

    def forward(self, block0_fmaps, block1_fmaps, block2_fmaps, block3_fmaps):
        batch_size = block0_fmaps.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        # (N, 8, 250, 250) 12 channels is for 2 boxes * (4 offsets)
        l_block0 = self.loc_block0(block0_fmaps)
        l_block0 = l_block0.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, 4)  # (N, 250*250*2, 4)

        l_block1 = self.loc_block1(block1_fmaps)  # (N, 8, 125, 125)
        l_block1 = l_block1.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, 4)  # (N, 125*125*2, 4)

        l_block2 = self.loc_block2(block2_fmaps)  # (N, 8, 63, 63)
        l_block2 = l_block2.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, 4)  # (N, 63*63*2, 4)

        l_block3 = self.loc_block3(block3_fmaps)  # (N, 8, 32, 32)
        l_block3 = l_block3.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, 4)  # (N, 32*32*2, 4)

        # Predict classes in localization boxes
        # (N, 2 * n_classes, 250, 250). 2 refers to 2 boxes per cell of fmap
        c_block0 = self.cl_block0(block0_fmaps)
        c_block0 = c_block0.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, self.n_classes)  # (N, 2*250*250, n_classes)

        c_block1 = self.cl_block1(block1_fmaps)  # (N, 2 * n_classes, 125, 125)
        c_block1 = c_block1.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, self.n_classes)  # (N, 2*125*125, n_classes)

        c_block2 = self.cl_block2(block2_fmaps)  # (N, 2 * n_classes, 63, 63)
        c_block2 = c_block2.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, self.n_classes)  # (N, 2*63*63, n_classes)

        c_block3 = self.cl_block3(block3_fmaps)  # (N, 2 * n_classes, 32, 32)
        c_block3 = c_block3.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, self.n_classes)  # (N, 2*32*32, n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([  # l_block0, l_block1,
            l_block2, l_block3], dim=1)
        classes_scores = torch.cat([  # c_block0, c_block1,
            c_block2, c_block3], dim=1)

        print("model.py: Break Point4")

        return locs, classes_scores


class SSD(nn.Module):
    def __init__(self, resnet_type, n_classes):
        super(SSD, self).__init__()
        assert resnet_type in [18, 34, 50]

        if resnet_type == 18:
            resnet = list(torchvision.models.resnet18().children())
            # output channels of fmap for each of blocks 0 - 3
            channels_for_block = [64, 128, 256, 512]
        elif resnet_type == 34:
            resnet = list(torchvision.models.resnet34().children())
            channels_for_block = [64, 128, 256, 512]
        else:
            resnet = list(torchvision.models.resnet50().children())
            channels_for_block = [256, 512, 1024, 2048]

        self.n_classes = n_classes
        self.pred_convs = PredictionConvolutions(channels_for_block, n_classes)
        self.pillar_feat_net = PFNv2()

        # Input channels of Conv2d in self.downsize must be output of PFN (N, 64, 500, 500)
        self.downsize = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                      nn.BatchNorm2d(
                                          64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(inplace=True))
        self.block0 = resnet[4]
        self.block1 = resnet[5]
        self.block2 = resnet[6]
        self.block3 = resnet[7]

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, lidar):

        encoded_bev = self.pillar_feat_net(lidar)
        x = self.downsize(encoded_bev)
        block0_fmaps = self.block0(x)
        block1_fmaps = self.block1(block0_fmaps)
        block2_fmaps = self.block2(block1_fmaps)
        block3_fmaps = self.block3(block2_fmaps)

        locs, classes_scores = self.pred_convs(block0_fmaps, block1_fmaps, block2_fmaps, block3_fmaps)

        print("model.py: Break Point5")

        return locs, classes_scores, encoded_bev

    def create_prior_boxes(self):

        fmap_dims = {  # 'block0': 250,
            # 'block1': 125,
            'block2': 63,
            'block3': 32}

        obj_scale = 0.031  # Assumes 500x500px BEV covers 80m x 80m and cars are 1.6m x 3.9m
        aspect_ratios = [2., 0.5]
        fmap_names = list(fmap_dims.keys())
        prior_boxes = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for fmap in fmap_names:
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios:
                        prior_boxes.append(
                            [cx, cy, obj_scale * sqrt(ratio), obj_scale / sqrt(ratio)])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        print("model.py: Break Point6")
        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size): # batch_size = 4 循环3次
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            # (n_priors, 4), these are fractional pt. coordinates
            # boundary coordinates (x_min, y_min, x_max, y_max)
            #print("decoded_locs: \n", decoded_locs) # torch.Size([9986, 4]) 9986 个 boxes

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)
            # print("S17: ", best_label)

            for c in range(1, self.n_classes): # n_classes = 2

                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c] # predicted_scores [3*9986*2]
                #print("class_scores:\n", class_scores) # 9986 个 分数， 每个box一个分类分数

                # torch.uint8 (byte) tensor, for indexing
                score_above_min_score = class_scores > min_score
                #print("score_above_min_score:\n", score_above_min_score)

                n_above_min_score = score_above_min_score.sum().item()
                # 3907 将 score 中所有值相加转成数字；意味着有 3907 项 True， 比最小分数大的有3907项


                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                # 保留class score 大于最小分数的那些box的分数
                # print("class_score_after:\n", class_scores)

                class_decoded_locs = decoded_locs[score_above_min_score]
                # 保留class score 大于最小分数的那些box的坐标
                # print("class_decoded_locs:\n", class_decoded_locs)


                # Sort predicted boxes and scores by scores from max to min
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]
                #print('S16 box_confidential_scores:\n', class_scores)
                #print('S16 box_maxmin_coordinates:\n', class_decoded_locs)
                # 按照上面已经排好顺序，第一个为置信度最高的那个框

                # Find the overlap between predicted boxes
                # overpal 第一行即为其他所有的候选框与置信度最高的那个框的IOU
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
                #print("S17 IOU: ", overlap)
                # 沿对角线对称的矩阵 （n1,n2）和 (n2,n1) 是一样的


                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                # suppress = torch.zeros(1, class_decoded_locs.size(0)).to(device)
                # print("S12: ", suppress)
                # for i in range(class_decoded_locs.size(0)):
                #     for j in range(class_decoded_locs.size(0)):
                #         overlap_box_boolen = (overlap[i][j] > max_overlap).type(torch.cuda.ByteTensor)# 对于 > 0.45 的所有值置为 1
                #         if overlap_box_boolen == 1:
                #             # overlap_box_boolen = torch.max(sup, overlap_box_boolen)  # (n_qualified)
                #             suppress.append(overlap_box_boolen)

                # print("S13: ", overlap[0] > max_overlap)
                # overlap_box_boolen = (overlap[0] > max_overlap).type(torch.cuda.ByteTensor)
                # # 对于 > 0.45(max_overlap) 的所有值置为 1, 其他值置为 0
                # print("S14: ", overlap_box_boolen)

                # Suppress = []
                #
                # for box in range(class_decoded_locs.size(0)):
                #     suppress = torch.zeros(1, class_decoded_locs.size(0)).to(device)
                #     suppress = torch.max(suppress, (overlap[box] > max_overlap).type(torch.cuda.ByteTensor))  # (n_qualified)
                #     Suppress.append(suppress)
                # # print("S15: ", Suppress)
                

                suppress = (overlap[0] > max_overlap).type(torch.cuda.ByteTensor)  # (n_qualified)
                
                #print("S18. suppress:", suppress)
                # suppress = torch.max(suppress,(overlap[box] > max_overlap).type(torch.cuda.ByteTensor))  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    # print("S19: ", suppress)
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    overlap_box = (overlap[box] > max_overlap).type(torch.cuda.ByteTensor)
                    #print(overlap_box)
                    #print(max_overlap)
                    
                    suppress = torch.max(suppress, overlap_box)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation
                    # 保留的box是和所有的IOU都不大于0.45的box

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                #print("S19. Suppress: ", suppress)

                # Store only unsuppressed boxes for this class
                #print("S20: ", class_decoded_locs.shape)
                #print("S21. ", class_decoded_locs[1 - suppress].shape)
                image_boxes.append(class_decoded_locs[1 - suppress]) # saved unsuppress is 1802 boxes
                # 保留的box是和所有的IOU都不大于0.45的box
                # 一个类保存为一个完整的tensor：[tensor1, tensor2]
                #print("S22. image_boxes: ", image_boxes)
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                #print("S23. image_labels: ", len(image_labels))
                image_scores.append(class_scores[1 - suppress])
                #print("S24. image_scores: ", len(image_scores))

            # If no object in any class is found, store a placeholder for 'background'
            # 上面设为List的格式就是为了方便检查是不是有类查到Box
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors 从list 再拆成 每一个类一个tensor
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            #print("S25.image_boxes", image_boxes)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0) # the number of saved boxes
            #print("n_objects", n_objects)

            # Keep only the top_k = 10 objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes(in tensor) and scores for all images
            all_images_boxes.append(image_boxes)
            #print("S26. predicted boxes: ", all_images_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        # lists of length batch_size
        print("model.py: Break Point7")
        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the prior boxes, a tensor of dimensions (N, n_priors, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, n_priors, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4),
                                dtype=torch.float).to(self.device)
        true_classes = torch.zeros(
            (batch_size, n_priors), dtype=torch.long).to(self.device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, n_priors)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(
                dim=0)  # (n_priors)

            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(
                range(n_objects)).to(self.device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            # (n_priors)
            label_for_each_prior = labels[i][object_for_each_prior]
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior <
                                 self.threshold] = 0  # (n_priors)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(
                boxes[i][object_for_each_prior]), self.priors_cxcy)  # (n_priors, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, n_priors)

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes),
                                           true_classes.view(-1))  # (N * n_priors)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, n_priors)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, n_priors)
        # (N, n_priors), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg[positive_priors] = 0.
        # (N, n_priors), sorted by decreasing hardness
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(
            0).expand_as(conf_loss_neg).to(self.device)  # (N, n_priors)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(
            1)  # (N, n_priors)
        # (sum(n_hard_negatives))
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()
                     ) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        print("model.py: Break Point8")

        return conf_loss + self.alpha * loc_loss
