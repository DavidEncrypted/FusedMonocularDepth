import os
import cv2
import torch
import numpy as np
import dill
import torchvision.transforms as transforms
from vnl.tools.parse_arg_test import TestOptions
from vnl.lib.models.metric_depth_model import MetricDepthModel
from vnl.lib.core.config import cfg, merge_cfg_from_file, print_configs
from vnl.lib.models.image_transfer import bins_to_depth
from argparse import Namespace
from tqdm import tqdm


class vnl_obj():
    def __init__(self, dataset="nyu"):
        print("VNL init")
        self.dataset = dataset
        if dataset == "nyu":
            test_args = Namespace(batchsize=2, cfg_file='lib/configs/resnext101_32x4d_nyudv2_class', dataroot='./', dataset='any', epoch=30, load_ckpt='./nyu_rawdata.pth', phase='test', phase_anno='test', results_dir='./evaluation', resume=False, start_epoch=0, start_step=0, thread=4, use_tfboard=False)
        elif dataset == "kitti":
            test_args = Namespace(batchsize=2, cfg_file='lib/configs/resnext101_32x4d_kitti_class', dataroot='./', dataset='kitti', epoch=30, load_ckpt='./kitti_eigen.pth', phase='test', phase_anno='test', results_dir='./evaluation', resume=False, start_epoch=0, start_step=0, thread=4, use_tfboard=False)

        test_args.thread = 1
        test_args.batchsize = 1

        merge_cfg_from_file(test_args)

        self.model = MetricDepthModel()

        self.model.eval()

        # load checkpoint
        if dataset == "nyu":
            load_ckpt("./vnl/nyu_rawdata.pth", self.model)
        elif dataset == "kitti":
            load_ckpt("./vnl/kitti_eigen.pth", self.model)

        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)


    def run(self, imgfilelist, basedatasetpath):
        pred_depths = []

        num_lines = sum(1 for line in open(imgfilelist,'r'))
        with open(imgfilelist, 'r') as f:
            for _, line in enumerate(tqdm(f, total=num_lines)):
                imgname = line.split()[0]
                fullpath = os.path.join(basedatasetpath, imgname)

                with torch.no_grad():
                    # Load image
                    img = cv2.imread(fullpath)
                    # Scale image
                    img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
                    img_torch = scale_torch(img_resize, 255)
                    img_torch = img_torch[None, :, :, :].cuda()
                    # Run model
                    _, pred_depth_softmax= self.model.module.depth_model(img_torch)
                    pred_depth = bins_to_depth(pred_depth_softmax)
                    pred_depth = pred_depth.cpu().numpy().squeeze()
                    # Store depth prediction and image path
                    if self.dataset == "kitti":
                        partial_image_path = os.sep.join(os.path.normpath(fullpath).split(os.sep)[-5:])
                    else:
                        partial_image_path = os.sep.join(os.path.normpath(fullpath).split(os.sep)[-2:])
                    outsample = {'pred_depth': pred_depth, 'image_path': partial_image_path}

                    pred_depths.append(outsample)
        return pred_depths


def load_ckpt(modelpath, model):
    """
    Load checkpoint.
    """
    if os.path.isfile(modelpath):
        #logger.info("loading checkpoint %s", modelpath)
        checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage, pickle_module=dill)
        model.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint
        torch.cuda.empty_cache()


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img
