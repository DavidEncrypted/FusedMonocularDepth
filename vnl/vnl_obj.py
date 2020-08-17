import os
import cv2
import torch
import numpy as np
import dill
#from lib.utils.net_tools import load_ckpt
#from vnl.lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from vnl.tools.parse_arg_test import TestOptions
#from data.load_dataset import CustomerDataLoader
from vnl.lib.models.metric_depth_model import MetricDepthModel
from vnl.lib.core.config import cfg, merge_cfg_from_file, print_configs
from vnl.lib.models.image_transfer import bins_to_depth
from argparse import Namespace
#logger = setup_logging(__name__)
from tqdm import tqdm


class vnl_obj():
    def __init__(self, dataset="nyu"):
        print("VNL init")

        if dataset == "nyu":
            test_args = Namespace(batchsize=2, cfg_file='lib/configs/resnext101_32x4d_nyudv2_class', dataroot='./', dataset='any', epoch=30, load_ckpt='./nyu_rawdata.pth', phase='test', phase_anno='test', results_dir='./evaluation', resume=False, start_epoch=0, start_step=0, thread=4, use_tfboard=False)
        elif dataset == "kitti":
            test_args = Namespace(batchsize=2, cfg_file='lib/configs/resnext101_32x4d_kitti_class', dataroot='./', dataset='kitti', epoch=30, load_ckpt='./kitti_eigen.pth', phase='test', phase_anno='test', results_dir='./evaluation', resume=False, start_epoch=0, start_step=0, thread=4, use_tfboard=False)
            print("RUNNING VNL KITTI")
        print("fake arg: ")
        print(test_args)

        test_args.thread = 1
        test_args.batchsize = 1

        merge_cfg_from_file(test_args)
        #print_configs(test_args)

        self.model = MetricDepthModel()

        self.model.eval()

        # load checkpoint
        # if self.test_args.load_ckpt:
        if dataset == "nyu":
            load_ckpt("./vnl/nyu_rawdata.pth", self.model)
        elif dataset == "kitti":
            load_ckpt("./vnl/kitti_eigen.pth", self.model)

        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)


    def run(self, imgfilelist, basedatasetpath):
        #print(imgfilelist)
        #print(basedatasetpath)

        pred_depths = []

        num_lines = sum(1 for line in open(imgfilelist,'r'))
        with open(imgfilelist, 'r') as f:
            for _, line in enumerate(tqdm(f, total=num_lines)):
                #print(line.split()[0])
                imgname = line.split()[0]
                #imgname = imgname.replace(".jpg", "W.jpg")
                fullpath = basedatasetpath + imgname
                #print(fullpath)

                with torch.no_grad():
                    img = cv2.imread(fullpath)
                    img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
                    img_torch = scale_torch(img_resize, 255)
                    img_torch = img_torch[None, :, :, :].cuda()

                    _, pred_depth_softmax= self.model.module.depth_model(img_torch)
                    pred_depth = bins_to_depth(pred_depth_softmax)
                    pred_depth = pred_depth.cpu().numpy().squeeze()

                    outsample = {'pred_depth': pred_depth, 'image_path': fullpath}

                    pred_depths.append(outsample)
        return pred_depths
            #pred_depth_scale = (pred_depth / pred_depth.max() * 60000).astype(np.uint16)  # scale 60000 for visualization


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


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    #logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = MetricDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    path = os.path.join(cfg.ROOT_DIR, '../../testset/imagestest') # the dir of imgs
    outpath = os.path.join(cfg.ROOT_DIR, '../../testset/imagesout/vnl') # the out dir of imgs
    realoutpath = os.path.join(cfg.ROOT_DIR, '../../testset/imagesout/vnl/realout') # the out dir of imgs
    gtpath = os.path.join(cfg.ROOT_DIR, '../../testset/imagesgt') # the out dir of imgs

    #path = os.path.join(cfg.ROOT_DIR, './test_any_imgs_examples') # the dir of imgs
    imgs_list = os.listdir(path)
    for i in imgs_list:
        #print(i)
        with torch.no_grad():
            img = cv2.imread(os.path.join(path, i))
            img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            _, pred_depth_softmax= model.module.depth_model(img_torch)
            pred_depth = bins_to_depth(pred_depth_softmax)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            pred_depth_scale = (pred_depth / pred_depth.max() * 60000).astype(np.uint16)  # scale 60000 for visualization


            cv2.imwrite(os.path.join(outpath, i.split('.')[0] + '-raw.png'), pred_depth_scale)
            #cv2.imwrite(os.path.join(path, i.split('.')[0] + '-raw.png'), pred_depth_scale)
