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
    # parser.add_argument('--bts_args_path', type=str, help='Path to bts args file', default='./bts_config/kitti_bts_arguments_partial_test_eigen.txt')
    parser.add_argument('--save_path', type=str, help='Path to the save dir', default='./data_save/')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset dir', default='/media/david/LinuxStorage/dataset/data_depth_annotated/test/')
        parser.add_argument('--filenames_path', type=str, help='Path to the filenames file', default='./bts_config/kitti_filenames_full_test.txt')
    parser.add_argument('--save_name', type=str, help='Appends \"_*save_name*\" to the model names in the save path', default='')



    args = parser.parse_args()


    #
    # #btsargs = './bts_config/kitti_bts_arguments_full_train_eigen.txt'
    # btsargs = './bts_config/kitti_bts_arguments_full_test_eigen.txt'
    #
    #
    # btsoutdir = './dataout/kitti/bts_out/'
    # vnloutdir = './dataout/kitti/vnl_out/'
    # meanoutdir = './dataout/kitti/mean_out/'
    #
    # savearraydir = '/media/david/LinuxStorage/data_storage/'

    bts = bts_obj(args.cfg_dir + 'kitti_bts_arguments_full_test_eigen.txt')
    vnl = vnl_obj("kitti")


    #np.set_printoptions(precision=3)

    # vnlfilenamesfile = "bts_config/kitti_filenames_full_test.txt"
    # #vnlfilenamesfile = "bts_config/kitti_filenames_partial_test.txt"
    #
    # #vnldatasetdir = "../dataset/nyu_depth_v2/official_splits/test/"
    # vnldatasetdir = "/media/david/LinuxStorage/dataset/data_depth_annotated/test/"

    # print("RUNNING BTS")
    btsoutarray = bts.run()
#    btsoutarray = []
    print("RUNNING VNL")
    vnloutarray = vnl.run(args.cfg_dir + 'kitti_filenames_full_test.txt', args.dataset_path)
    #vnloutarray = vnl.run("bts_config/kitti_filenames_partial_test.txt", args.dataset_path)
    #vnloutarray = []
    # saving bts and vnl array

    print("bts shape: ", btsoutarray[0]['pred_depth'].shape)
    print("vnl shape: ", vnloutarray[0]['pred_depth'].shape)
    for vnlsample in vnloutarray:
        #print("pre shape", vnlsample['pred_depth'].shape)
        top_margin = int(vnlsample['pred_depth'].shape[0] - 352)
        left_margin = int((vnlsample['pred_depth'].shape[1] - 1216) / 2)
        vnlsample['pred_depth'] = vnlsample['pred_depth'][top_margin:top_margin + 352, left_margin:left_margin+1216]
        print("post shape", vnlsample['pred_depth'].shape)

    print("Saving arrays")
    # np.save(savearraydir + 'btsout_kitti_full_train.npy', btsoutarray)
    # np.save(savearraydir + 'vnlout_kitti_full_test.npy', vnloutarray)



    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Saving outputs
    if args.save_name == '':
        np.save(args.save_path + 'kitti_bts.npy', btsoutarray)
        np.save(args.save_path + 'kitti_vnl.npy', vnloutarray)
    else:
        np.save(args.save_path + 'kitti_bts_' + args.save_name + '.npy', btsoutarray)
        np.save(args.save_path + 'kitti_vnl_' + args.save_name + '.npy', vnloutarray)
