#  python3 convert.py /home/redhwan/2/HPE/WQuatNet/output/snapshots/CMU_20240108115256_bs40/CMU_epoch_51.tar /home/redhwan/2/HPE/quat1/pretrained_cmu/geodesic_loss/CMU_epoch_51.pth

import argparse

import torch

from backbone.repvgg import repvgg_model_convert
from sixdrepnet.model import SixDRepNet,  RotationNet, Quat, WQuatNet

parser = argparse.ArgumentParser(description='SixDRepNet Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-D2se')


def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def convert():
    args = parser.parse_args()

    print('Loading model.')
    model = WQuatNet(backbone_name=args.arch,
                        backbone_file='',
                        deploy=False,
                        pretrained=False)


    # Load snapshot
    saved_state_dict = torch.load(args.load)

    load_filtered_state_dict(model, saved_state_dict['model_state_dict'])
    print('Converting model.')
    repvgg_model_convert(model, save_path=args.save)
    print('Done.')

if __name__ == '__main__':
    convert()