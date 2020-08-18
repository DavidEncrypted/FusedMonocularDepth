import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sharpnet.resnet import Bottleneck as ResBlock
from sharpnet.sharpnet_model import *
from PIL import Image
from sharpnet.data_transforms import *
import os, sys
from imageio import imread, imwrite
from tqdm import tqdm

# GLOBAL ARGS
model_path = "./sharpnet/models/final_checkpoint_NYU.pth"
rescale_factor = 1
cuda_device = 0

def round_down(num, divisor):
    return num - (num % divisor)

class sharpnet_obj():
    def __init__(self, dataset="nyu"):
        self.dataset = dataset
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        self.device = torch.device("cuda" if cuda_device != '' else "cpu")
        print("Running on " + torch.cuda.get_device_name(self.device))


        self.model = SharpNet(ResBlock, [3, 4, 6, 3], [2, 2, 2, 2, 2],
                         use_normals=False,
                         use_depth=True,
                         use_boundary=False,
                         bias_decoder=True)

        torch.set_grad_enabled(False)

        model_dict = self.model.state_dict()

        # Load model
        trained_model_path = model_path
        trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)

        # load image resnet encoder and mask_encoder and normals_decoder (not depth_decoder or normal resnet)
        model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}

        self.model.load_state_dict(model_weights)
        self.model.eval()
        self.model.to(self.device)

        scale = rescale_factor

        mean_RGB = np.array([0.485, 0.456, 0.406])
        mean_BGR = np.array([mean_RGB[2], mean_RGB[1], mean_RGB[0]])

    def run(self, imgfilelist, basedatasetpath, white):
        pred_depths = []
        num_lines = sum(1 for line in open(imgfilelist,'r'))
        with open(imgfilelist, 'r') as f:
            for _, line in enumerate(tqdm(f, total=num_lines)):
                imgname = line.split()[0]
                # Remove W from iamge name
                if white:
                    imgname = imgname.replace(".jpg", "W.jpg")
                fullpath = os.path.join(basedatasetpath, imgname)

                with torch.no_grad():
                    # Load image
                    image_pil = Image.open(fullpath)
                    w, h = image_pil.size
                    # Run model
                    pred_depth = self.get_pred_from_input(image_pil)

                    if self.dataset == "kitti":
                        partial_image_path = os.sep.join(os.path.normpath(fullpath).split(os.sep)[-5:])
                    else:
                        partial_image_path = os.sep.join(os.path.normpath(fullpath).split(os.sep)[-2:])
                    outsample = {'pred_depth': pred_depth, 'image_path': partial_image_path}

                    pred_depths.append(outsample)
        return pred_depths

    def get_pred_from_input(self, image_pil):
        normals = None
        boundary = None
        depth = None

        image_np = np.array(image_pil)
        w, h = image_pil.size

        scale = rescale_factor

        h_new = round_down(int(h * scale), 16)
        w_new = round_down(int(w * scale), 16)

        if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
            print("Input image has only 1 channel, please use an RGB or RGBA image")
            sys.exit(0)

        if len(image_np.shape) == 4 or image_np.shape[-1] == 4:
            # RGBA image to be converted to RGB
            image_pil = image_pil.convert('RGBA')
            image = Image.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
            image.paste(image_pil.copy(), mask=image_pil.split()[3])
        else:
            image = image_pil

        image = image.resize((w_new, h_new), Image.ANTIALIAS)

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t = []
        t.extend([ToTensor(), normalize])
        transf = Compose(t)

        data = [image, None]
        image = transf(*data)

        image = torch.autograd.Variable(image).unsqueeze(0)

        image = image.to(self.device)

        depth_pred = self.model(image)

        tmp = depth_pred.data.cpu()

        shp = tmp.shape[2:]

        mask_pred = np.ones(shape=shp)
        mask_display = mask_pred

        depth_pred = depth_pred.data.cpu().numpy() * 65535 / 1000

        return depth_pred[0][0]
