# python3 test.py  --batch_size 64  --dataset AFLW2000  --data_dir /home/redhwan/2/data/AFLW2000  --filename_list /home/redhwan/2/data/AFLW2000/files.txt --snapshot /home/redhwan/2/HPE/WQuatNet/pretrained_300w_D2se/quaternion_angular_error_loss_eps/300W_LP_epoch_120.pth  --show_viz False
import time
import math
import re
import sys
import os
import argparse
from scipy.spatial.transform import Rotation
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')

from model import WQuatNet
import utils
import datasets
import  datasets3
from scipy.spatial.transform import Rotation as R

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir',
                        dest='data_dir', help='Directory path for data.',
                        default='/home/redhwan/2/data/AFLW2000',
                        # default='datasets/AFLW2000',
                        type=str)
    parser.add_argument('--filename_list',
                        dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='/home/redhwan/2/data/AFLW2000/files.txt'
                        # default = 'datasets/AFLW2000/files.txt'
                        , type=str)  # datasets/BIWI_noTrack.npz
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='model.pth',
                        # default='pretrained/6DRepNet_300W_LP_AFLW2000.pth',
                        type=str)
    parser.add_argument('--batch_size',
                        dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--dataset',
                        dest='dataset', help='Dataset type.',
                        default='AFLW2000', type=str)


    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    # batch_size = args.batch_size
    model = WQuatNet(backbone_name='RepVGG-D2se',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)


    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(
                                              224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets3.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2)


    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    total, total_front = 0, 0
    yaw_error = pitch_error = roll_error = .0
    yaw_error_front = pitch_error_front = roll_error_front = .0
    v0_errq = v1_errq = v2_errq = v3_errq = .0
    v1_err = v2_err = v3_err = .0

    with torch.no_grad():
        time_list = []
        for i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).cuda(gpu)
            total += cont_labels.size(0)


            R_gt = r_label
            R_gt_Q = r_label
            R_gt = utils.convert_quaternions_to_rotation_matrices(R_gt)
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            R_pred = model(images)
            matrices =  utils.convert_quaternions_to_rotation_matrices(R_pred)
            euler = utils.compute_euler_angles_from_rotation_matrices(matrices, full_range=True) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()
            start_time = time.time()

            for j in range(len(y_gt_deg)):
                pitch_error_temp = torch.sum(torch.min(torch.stack((
                    torch.abs(p_gt_deg[j] - p_pred_deg[j]),
                    torch.abs(p_pred_deg[j] + 360 - p_gt_deg[j]),
                    torch.abs(p_pred_deg[j] - 360 - p_gt_deg[j]),
                    torch.abs(p_pred_deg[j] + 180 - p_gt_deg[j]),
                    torch.abs(p_pred_deg[j] - 180 - p_gt_deg[j]))), 0)[0])
                yaw_error_temp = torch.sum(torch.min(torch.stack((
                    torch.abs(y_gt_deg[j] - y_pred_deg[j]),
                    torch.abs(y_pred_deg[j] + 360 - y_gt_deg[j]),
                    torch.abs(y_pred_deg[j] - 360 - y_gt_deg[j]),
                    torch.abs(y_pred_deg[j] + 180 - y_gt_deg[j]),
                    torch.abs(y_pred_deg[j] - 180 - y_gt_deg[j]))),
                    0)[0])
                roll_error_temp = torch.sum(torch.min(torch.stack((
                    torch.abs(r_gt_deg[j] - r_pred_deg[j]),
                    torch.abs(r_pred_deg[j] + 360 - r_gt_deg[j]),
                    torch.abs(r_pred_deg[j] - 360 - r_gt_deg[j]),
                    torch.abs(r_pred_deg[j] + 180 - r_gt_deg[j]),
                    torch.abs(r_pred_deg[j] - 180 - r_gt_deg[j]))), 0)[0])

                pitch_error += pitch_error_temp
                yaw_error += yaw_error_temp
                roll_error += roll_error_temp



                if abs(y_gt_deg[j]) < 90:
                    total_front += 1
                    # print('total_front:', total_front)
                    pitch_error_front += pitch_error_temp
                    yaw_error_front += yaw_error_temp
                    roll_error_front += roll_error_temp

            if args.show_viz:
                name = name[0]
                if args.dataset == 'AFLW2000':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))

                elif args.dataset == 'BIWI':
                    vis = np.uint8(name)
                    h, w, c = vis.shape
                    vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
                    vis0 = cv2.fromarray(vis)
                    cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)
                    cv2_img = cv2.imread(vis2)
                utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx=200, tdy=200, size=100)
                # utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=200)
                cv2.imshow("Test", cv2_img)
                cv2.waitKey(5)
                cv2.imwrite(os.path.join('output/img/', name + '.png'), cv2_img)

        # print("Inference time per image: ", sum(time_list) / len(time_list))

        print('[Total heads: %d] Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (total,
                                                                                   yaw_error / total,
                                                                                   pitch_error / total,
                                                                                   roll_error / total,
                                                                                   (
                                                                                               yaw_error + pitch_error + roll_error) / (
                                                                                               total * 3)))



        if total_front != 0:
            print('[Front faces: %d] Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (total_front,
                                                                                       yaw_error_front / total_front,
                                                                                       pitch_error_front / total_front,
                                                                                       roll_error_front / total_front,
                                                                                       (
                                                                                                   yaw_error_front + pitch_error_front + roll_error_front) / (
                                                                                                   total_front * 3)))



