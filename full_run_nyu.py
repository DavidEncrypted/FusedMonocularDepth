from bts.bts_obj import bts_obj
from vnl.vnl_obj import vnl_obj
from sharpnet.sharpnet_obj import sharpnet_obj
import cv2
import numpy as np
import sys
import argparse
import os


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Fuse NYU run')

    parser.add_argument('--cfg_dir', type=str, help='Path to config dir', default='./bts_config/')
    parser.add_argument('--save_path', type=str, help='Path to the save dir', default='./data_save/')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset dir', default='./datasets/nyu/')
    parser.add_argument('--filenames_path', type=str, help='Path to the filenames file', default='bts_config/bts_filenames_full_test.txt')
    parser.add_argument('--save_name', type=str, help='Appends \"_*save_name*\" to the model names in the save path', default='')

    # '../dataset/nyu_depth_v2/official_splits/test2/'

    args = parser.parse_args()

    # Creating pytorch model objects
    sharpnet = sharpnet_obj()
    print(args.cfg_dir + "bts_arguments_full_test.txt")
    bts = bts_obj(args.cfg_dir + "bts_arguments_full_test.txt", args.filenames_path, os.path.join(args.dataset_path, "test"))
    vnl = vnl_obj()

    # Running pytorch model objects on the images in the filenames file

    print("RUNNING BTS")
    btsoutarray = bts.run()

    print("RUNNING SHARPNET")
    sharpnetoutarray = sharpnet.run(args.filenames_path, os.path.join(args.dataset_path, "test"), True)



    print("RUNNING VNL")
    vnloutarray = vnl.run(args.filenames_path, os.path.join(args.dataset_path, "test"))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Saving outputs
    if args.save_name == '':
        np.save(args.save_path + 'nyu_sharpnet.npy', sharpnetoutarray)
        np.save(args.save_path + 'nyu_bts.npy', btsoutarray)
        np.save(args.save_path + 'nyu_vnl.npy', vnloutarray)
    else:
        np.save(args.save_path + 'nyu_sharpnet_' + args.save_name + '.npy', sharpnetoutarray)
        np.save(args.save_path + 'nyu_bts_' + args.save_name + '.npy', btsoutarray)
        np.save(args.save_path + 'nyu_vnl_' + args.save_name + '.npy', vnloutarray)
