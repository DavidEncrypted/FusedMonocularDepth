# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts.bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts.bts import BtsModel


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class bts_obj():
    def __init__(self, arguments_filename, filenames_path, dataset_path, dataset="nyu"):
        print("BTS init")
        self.dataset = dataset
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = convert_arg_line_to_args

        parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
        parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                            default='densenet161_bts')
        parser.add_argument('--data_path', type=str, help='path to the data', required=False)
        parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=False)
        parser.add_argument('--input_height', type=int, help='input height', default=480)
        parser.add_argument('--input_width', type=int, help='input width', default=640)
        parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
        parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
        parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
        parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
        parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
        parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)


        arg_filename_with_prefix = '@' + arguments_filename
        self.args = parser.parse_args([arg_filename_with_prefix])

        print(self.args.data_path)
        self.args.data_path = dataset_path
        self.args.filenames_file = filenames_path
        print(self.args.data_path)
        print(self.args.filenames_file)

        model_dir = os.path.dirname(self.args.checkpoint_path)
        sys.path.append(model_dir)

        for key, val in vars(__import__(self.args.model_name)).items():
            if key.startswith('__') and key.endswith('__'):
                continue
            vars()[key] = val

        self.dataloader = BtsDataLoader(self.args, 'test')


        self.model = BtsModel(params=self.args)
        self.model = torch.nn.DataParallel(self.model)

        checkpoint = torch.load(self.args.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.cuda()


    def run(self):

        # image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        # focal = 518.8579
        pred_depths = []
        with torch.no_grad():
            for _, sample in enumerate(tqdm(self.dataloader.data)):
                image = Variable(sample['image'].cuda())
                focal = Variable(sample['focal'].cuda())
                image_path = sample['image_path'][0]
                #print(image_path)
                _, _, _, _, depth_est = self.model(image, focal)
                #print(os.sep.join(os.path.normpath(image_path).split(os.sep)[-5:]))
                if self.dataset == "kitti":
                    partial_image_path = os.sep.join(os.path.normpath(image_path).split(os.sep)[-5:])
                else:
                    partial_image_path = os.sep.join(os.path.normpath(image_path).split(os.sep)[-2:])
                outsample = {'pred_depth': depth_est.cpu().numpy().squeeze(), 'image_path': partial_image_path}
                #pred_depths.append(depth_est.cpu().numpy().squeeze())
                pred_depths.append(outsample)
            # image = self.to_tensor(image)
            # print("image to_tensor shape: ", image.shape)
            # image = self.normalize(image)
            #
            # image = Variable(image.cuda())
            # #focal = Variable(focal.cuda())
            # print("image tensor shape: ", image.shape)
            # print("image tensor: ", image)
            # _, _, _, _, depth_est = self.model(image, focal)
            #
            # depth_est = depth_est.cpu().numpy().squeeze()

        return pred_depths
