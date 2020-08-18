from bts.bts_obj import bts_obj
from vnl.vnl_obj import vnl_obj
import cv2
import numpy as np
import sys
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuse KITTI run')

    parser.add_argument('--cfg_dir', type=str, help='Path to config dir', default='./bts_config/')
    parser.add_argument('--save_path', type=str, help='Path to the save dir', default='./data_save/')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset dir', default='./datasets/kitti/data_depth_annotated/')
    parser.add_argument('--filenames_path', type=str, help='Path to the filenames file', default='./bts_config/kitti_filenames_full_test.txt')
    parser.add_argument('--save_name', type=str, help='Appends \"_*save_name*\" to the model names in the save path', default='')

    # '/media/david/LinuxStorage/dataset/data_depth_annotated/'

    args = parser.parse_args()

    # Create model objects
    bts = bts_obj(args.cfg_dir + 'kitti_bts_arguments_full_test_eigen.txt', args.filenames_path, os.path.join(args.dataset_path, "test"), "kitti")
    vnl = vnl_obj("kitti")

    print("RUNNING VNL")
    vnloutarray = vnl.run(os.path.join(args.cfg_dir, 'kitti_filenames_full_test.txt'), os.path.join(args.dataset_path, "test"))

    print("RUNNING BTS")
    btsoutarray = bts.run()


    # KB Cropping the vnl samples
    for vnlsample in vnloutarray:
        top_margin = int(vnlsample['pred_depth'].shape[0] - 352)
        left_margin = int((vnlsample['pred_depth'].shape[1] - 1216) / 2)
        vnlsample['pred_depth'] = vnlsample['pred_depth'][top_margin:top_margin + 352, left_margin:left_margin+1216]

    print("Saving arrays")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Saving outputs
    if args.save_name == '':
        np.save(args.save_path + 'kitti_bts.npy', btsoutarray)
        np.save(args.save_path + 'kitti_vnl.npy', vnloutarray)
    else:
        np.save(args.save_path + 'kitti_bts_' + args.save_name + '.npy', btsoutarray)
        np.save(args.save_path + 'kitti_vnl_' + args.save_name + '.npy', vnloutarray)
