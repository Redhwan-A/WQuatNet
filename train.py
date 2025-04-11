
# python3 train.py --dataset AFLW2000 --data_dir /home/redhwan/2/data/AFLW2000 --filename_list /home/redhwan/2/data/AFLW2000/files.txt  --output_string AFLW2000
import time
import datetime
import math
import re
import sys
import os
import argparse
import csv
import numpy as np
from numpy.lib.function_base import _quantile_unchecked
# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
# from face_detection import RetinaFace
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

matplotlib.use('TkAgg')

from model import  WQuatNet
import utils
import datasets3
# from sixdrepnet.loss import GeodesicLoss
from losses_quat import  (quaternion_distance_loss, quaternion_angular_error_loss,quaternion_angular_error_loss_eps, lie_algebra_loss, opal_loss,
                          unit_quaternion_regularization_loss, quaternion_loss, quaternion_mse_loss, quaternion_geodesic_loss_Red_epilon,
                          quaternion_Anti_Geodesic_loss, quaternion_geodesic_loss,  bingham_loss, bingham_likelihood_loss, quat_squared_loss, quat_squ_loss,
                          frobenius_squared_norm_loss, rotmat_frob_squared_norm_loss, quat_chordal_squared_loss, quat_loss,
                          quat_consistency_loss, quat_self_supervised_primal_loss, quat_geodesic_loss_antipodal, quaternion_geodesic_loss_Red,
                          angle_axis_loss, l2_ordinal_loss, wrapped_loss, quaternion_angular_error_loss_eps_min, quaternion_angular_error_loss_eps_abs)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Train a deep network to predict 3D expression.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=800, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, #40
        # default=16,
        type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float) #0.00001
    parser.add_argument('--scheduler',
                        default=False,
                        # default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='AFLW2000', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='data/AFLW2000', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='data/AFLW2000/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='',
        type=str)

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
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler
    dataset_name = args.dataset
    snapshot_name = args.snapshot
    # =====================learn_info tar ==================
    datetime_ = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    time_ = time.time()
    print("datetime_",datetime_,"time_",time_)

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots') #Redhwan added exist_ok=True

    summary_name = '{}_{}_bs{}'.format(
        dataset_name, datetime_, args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name)) #Redhwan added exist_ok=True
    #=====================learn_info txt==================
    if not os.path.exists('output/learn_info'):
        os.makedirs('output/learn_info') #Redhwan added exist_ok=True

    name_txt = '{}_{}'.format(
        dataset_name, datetime_)

    if not os.path.exists('output/learn_info/{}'.format(name_txt)):
        os.makedirs('output/learn_info/{}'.format(name_txt)) #Redhwan added exist_ok=True

    # Define the Google Drive file ID of the model snapshot
    # url = "https://drive.google.com/file/d/1wzE_7zDEP62jq6PI9b8CFovEQz190M4U/view?usp=drive_link"
    # backbone_file = "RepVGG-D2se-200epochs-train.pth"
    #
    # gdown.download(url, backbone_file, use_cookies=False)
    # checkpoint = torch.load('RepVGG-D2se-200epochs-train.pth', map_location='cpu')
    # print(checkpoint)

    backbone_name = 'RepVGG-D2se'
    # backbone_name = 'RepVGG-B1g2'
    backbone_file = 'RepVGG-D2se-200epochs-train.pth'
    # backbone_file = 'RepVGG-B1g2-train.pth'

    model = WQuatNet(backbone_name,
                       backbone_file,
                       deploy=False,
                       pretrained=True)

    print ("model", model)
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
#224, 256,128
    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets3.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    print('pose_dataset_____________', pose_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    crit = quaternion_angular_error_loss_eps_abs
    optimizer = torch.optim.Adam(model.parameters(), args.lr) #Adam
    print('optimizer', optimizer)


    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    outfile = open('output/learn_info/' + name_txt + '/' + args.output_string + '.txt', "a+")
    outfileplot = open('output/learn_info/' + name_txt + '/' + args.output_string + 'plot.txt', "a+")
    b= ('optimizer: %s, crit: %s, dataset_name: %s, backbone_name: %s, backbone_file: %s,  batch_size: %d , lr: ' '%.7f' % ( optimizer, crit, dataset_name, backbone_name,backbone_file,batch_size, args.lr)  )
    outfile.write('\n')
    outfile.write(b)
    outfileplot.write(b)
    min_loss = 9999
    start_time = time.time()
    checkpoint_interval = 5
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0

        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            # print(images.shape)
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)
            loss = crit(labels.cuda(gpu), pred_mat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()


            if (i+1) % int(len(train_loader)//3) == 0: #if (i+1) % 100 == 0:
                a= ('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.7f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )
                print(a)

                outfile.write('\n')
                outfile.write(a)

            # Checkpoint for every 10 epochs
        if epoch % checkpoint_interval == 0:
            interval_time = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} completed in {interval_time:.2f} seconds")
            start_time = time.time()  # Reset the timer for the next interval
        avg_loss = loss_sum / (i + 1)
        if min_loss > avg_loss:
            min_loss = avg_loss


        b = ("Epoch: %d, avg_loss: %.7f, min_loss: %.7f" % (epoch + 1, avg_loss, min_loss))
        c = ('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.7f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          avg_loss,
                      )
                      )
        print(b)
        outfile.write('\n')
        outfile.write(b)
        outfileplot.write('\n')
        outfileplot.write(c)
        if b_scheduler:
            print('kkkkkkkkkkkkkkkkkk')
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.tar')
                  )

    outfile.close()
    outfileplot.close()


