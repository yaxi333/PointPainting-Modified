from utils import *
from dataset import KittiDataset
import torch
import numpy as np
import os
import pickle
from model import SSD
import tensorflow

""" 配置GPU以及索引 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
""" 设置训练模型的保存参数的位置 """
checkpoint = './pointpillars.pth'
checkpoint = torch.load(checkpoint, map_location='cpu')
""" 配置评估模型，并将上一步加载的模型的训练权重配置到SSD中 """
model = SSD(resnet_type=34, n_classes=2) # model.py: Break Point 3
model.load_state_dict(checkpoint)
model = model.to(device)
""" root = /Users/xymbiotec/Desktop/my_point_painting-main/work"""

root = "/content/drive/My Drive/Colab Notebooks/pointpainting/work"
results = root + 'results'
if not os.path.exists(results):
    os.makedirs(results)

""" 配置训练 """
bev_len = 500
ymax_cam, ymin_cam = 1.21, 2.91
h = ymin_cam - ymax_cam
z = np.abs((ymax_cam - ymin_cam) / 2)
theta = np.pi/2
cam_to_img = np.array([[7.25995079e+02,  9.75088160e+00,  6.04164953e+02, 4.48572807e+01],
                       [-5.84187313e+00,  7.22248154e+02, 1.69760557e+02, 2.16379106e-01],
                       [7.40252715e-03,  4.35161404e-03,  9.99963105e-01, 2.74588400e-03]])

with open('file_nums_with_cars.pkl', 'rb') as f:
    file_nums_with_cars = pickle.load(f)

print("eval.py: Break Point3")

def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """
    print("eval.py: Break Point4")
    # Make sure it's in eval mode
    model.eval()
    # model.run()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    print("eval.py: Break Point5")

    with torch.no_grad():

        for i, augmented_lidar_cam_coords in enumerate(test_loader):
            print("Epoch #",i)

            #print("mod1:  ", augmented_lidar_cam_coords)

            # tuple transfer to list, each list has 3 big tensor components like x,y,z, dimension [not sure]
            # augmented_lidar_cam_coords = list(tuple(augmented_lidar_cam_coords))
            augment_whole = []

            # print("mod:  ", augmented_lidar_cam_coords[0])
            for _, augment in enumerate(augmented_lidar_cam_coords):
                aug_array = []
                for _, aug_tensor in enumerate(augment):
                    #print("S1 =", aug_tensor)
                    #print("S2 =", np.array(aug_tensor.cpu()))
                    aug_tensor = np.array(aug_tensor.cpu())
                    aug_array.append(aug_tensor)
                #print("S3 =", aug_array)
                augment_whole.append(aug_array)
            #print("S4 =", augment_whole)

            # augmented_lidar_cam_coords = torch.tensor(augment_whole, device=device)

            #augmented_lidar_cam_coords = augmented_lidar_cam_coords.to(device)

            # print("S5 =", augmented_lidar_cam_coords)
            print("eval.py: Break Point6")
            predicted_locs, predicted_scores, _ = model(augment_whole)

            print("eval.py: Break Point7")
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,min_score=0.2, max_overlap=0.45, top_k=10)
            print("eval.py: Break Point8")
            #boxes = [b.to(device) for b in det_boxes_batch]
            #print("S1:\n", boxes)
            # 为什么要乘以 bev_len? 3维变成 ——> 1500=3*500 维
            boxes = [b.to(device) for b in det_boxes_batch] * bev_len
            #print("S2. boxes: ", boxes)
            label = [l.to(device) for l in det_labels_batch]
            #print("S2. labels:\n ", label)
            score = [s.to(device) for s in det_scores_batch]
            #print("S3. score:\n ", score)
            xmin, ymin, xmax, ymax = boxes[0][:,0], boxes[0][:,1], boxes[0][:,2], boxes[0][:,3]
            xmin, ymin, xmax, ymax = np.array(xmin.cpu()), np.array(ymin.cpu()), np.array(xmax.cpu()), np.array(ymax.cpu())
            # 求前10个返回来的box位置及宽度的均值：
            xmin, ymin, xmax, ymax = np.mean(xmin),np.mean(ymin),np.mean(xmax),np.mean(ymax)
            #print("S4. xmin:\n", xmin)
            x = (xmax - xmin) / 2
            y = (ymax - ymin) / 2
            #print("S5. x coord of center point of each box\n ", x)
            #print("S6. y coord of center point of each box\n ", y)

            w, l = xmax - xmin, ymax - ymin
            #print("S7. w:\n ", w)
            #print("S8. l: \n", l)
   

            if w.any() > l.any():
                w, l, theta = l, w, 0
                return print("Error: some width is larger than length! ")
            else:
                theta = np.pi/2
            # print("S7. w:\n ", w)
            # print("S8. l: \n", l)

            # cam_points = np.transpose([[xmin, ymin_cam, z, 1], [xmax, ymax_cam, z, 1]])
            cam_points = np.transpose([xmin, ymin_cam, z, 1])
            #print("S9. came_points:\n", cam_points)
            #print("S9 end")
            img_points = cam_to_img.dot(cam_points).T
            #print("S10 imag_points:\n", img_points)
            f_name = file_nums_with_cars[i] + '.txt'

            print("eval.py: Break Point9")
            with open(f_name, "w+") as f:
                # f.write(f"Car 0 0 0 {img_points[0]}, {img_points[1]}, {img_points[2]} ,{img_points[3]} ,{h} ,{w} ,{l}, {x}. {y}, {z}, {theta}, {score}")
                f.write(f"Car 0 0 0 {img_points[0]}, {img_points[1]}, {img_points[2]},{h} ,{w} ,{l}, {x}. {y}, {z}, {theta}, {score}")


def test_data_prepare(root, mode='testing', valid=False, batch_size=4):
    test_dataset = KittiDataset(root, mode, valid=False)
    test_datalodaer = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                  collate_fn=test_dataset.collate_fn)
    print("eval.py: Break Point1")
    return test_datalodaer

def dup_remove_set(list_raw):
    result = set()
    for sublist in list_raw:
        item = set(sublist)
        result = result.union(item)

    print("eval.py: Break Point2")
    return list(result)


evaluate(test_data_prepare(root), model)
