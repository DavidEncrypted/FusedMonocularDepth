import cv2
import numpy as np
import sys
from scipy.optimize import minimize
from astropy.stats import sigma_clip
import argparse
from compute_eval import *
import os

eigen_crop = True
min_depth_eval = 1e-3
max_depth_eval = 10
dataset = "nyu"

# Evaluate predicted depths using only one metric
def eval_met(pred_depths, metric, gt): # metric: 1 = rmse, 2 = d1, 3 = log10
    num_samples = len(pred_depths)
    met = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt[i]
        pred_depth = pred_depths[i]

        # Make sure predicted depth is within constraints
        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        if eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        # Use chosen metric
        if metric == 1:
            met[i] = compute_rmse(gt_depth[valid_mask], pred_depth[valid_mask])
        elif metric == 2:
            met[i] = compute_d1(gt_depth[valid_mask], pred_depth[valid_mask])
        elif metric == 3:
            met[i] = compute_log10(gt_depth[valid_mask], pred_depth[valid_mask])
        else:
            met[i] = 0
    return met

# Evaluate the predicted depths using ground truth
def eval(pred_depths, gt):

    num_samples = len(pred_depths)
    pred_depths_valid = []

    i = 0
    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        if eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), log10.mean()))

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3

# Load the ground truth using the data in the outarray.
def loadgt(outarray, config_dir, ds_dir):
    missing_ids = 0
    gt_depths = []

    filenames_train = os.path.join(config_dir,"bts_filenames_full_test_train.txt")
    filetrain = []
    with open(filenames_train, 'r') as f:
        for line in f:
            filetrain.append(line.split()[0])

    outarray_train = []
    outarray_val = []

    for sample in outarray:
        path = sample['image_path']
        gt_depth_path = os.path.join(ds_dir, "test", os.path.split(path)[0], "sync_depth_" + os.path.split(path)[1][4:-4] + ".png")
        name = sample['image_path']

        depth = cv2.imread(gt_depth_path, -1)
        if depth is None:
            print('Missing: %s ' % gt_depth_path)
            missing_ids += 1
            continue
        depth = depth.astype(np.float32) / 1000.0

        if name in filetrain:
            outarray_train.append(depth)
        else:
            outarray_val.append(depth)

        gt_depths.append(depth)

    print("num gt: ", len(gt_depths))
    print("num GT missing: ", missing_ids)
    return gt_depths, outarray_train, outarray_val

# Combine the input models according to the weights in y
# The choice contains a mask to chose which models to use
# Output the calculated metrics
def print_array(y, choice, bts, vnl, sn, gt):
    x = np.array([0.0,0.0,0.0])
    if len(y) == 2:
        i = 0
        j = 0
        for val in choice:
            if val == 1:
                x[i] = y[j]
                j += 1
            i += 1

    if len(y) == 3:
        i = 0
        #j = 0
        for val in choice:
            if val == 1:
                x[i] = y[i]
                #j += 1
            i += 1

    calcoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(bts, vnl, sn):
        btsdepth = btssample['pred_depth']
        #print(btsdepth)
        vnldepth = vnlsample['pred_depth']
        # VNL outputs depth that needs to be scaled by 10 to be compatible with BTS outputted depth
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean3(btsdepth, vnldepth, sharpnetdepth, x[0], x[1], x[2])
        calcoutarray.append(calcdepth)

    _,_,_,_,rms,_,_,_,_ = eval(calcoutarray, gt)
    print(x[0], " * BTS, ", x[1], " * VNL,", x[2], " * SHARPNET | RMSE: ", rms.mean(), "\n")

# Combine the input models according to the weights in x
# Output the calculated metrics
def eval_array(x, bts, vnl, sn, gt):
    calcoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(bts, vnl, sn):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        # VNL outputs depth that needs to be scaled by 10 to be compatible with BTS outputted depth
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean3(btsdepth, vnldepth, sharpnetdepth, x[0], x[1], x[2])
        calcoutarray.append(calcdepth)

    met = eval_met(calcoutarray, 1, gt) # rmse
    return met.mean()

# Combine the input models according to the weights in y
# The choice contains a mask to chose which models to use
# Output the calculated metrics
def eval_two_array(y, choice, bts, vnl, sn, gt):

    x = np.array([0.0,0.0,0.0])
    i = 0
    j = 0
    for val in choice:
        if val == 1:
            x[i] = y[j]
            j += 1
        i += 1

    calcoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(bts, vnl, sn):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        # VNL outputs depth that needs to be scaled by 10 to be compatible with BTS outputted depth
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean3(btsdepth, vnldepth, sharpnetdepth, x[0], x[1], x[2])
        calcoutarray.append(calcdepth)

    met = eval_met(calcoutarray, 1, gt)
    return met.mean()

# Combine the input models using a median
# Output the calculated metrics
def eval_median(bts, vnl, sn, gt):
    calcoutarray = []
    pathoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(bts, vnl, sn):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        # VNL outputs depth that needs to be scaled by 10 to be compatible with BTS outputted depth
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = np.median([btsdepth, vnldepth, sharpnetdepth], axis=0)
        pathoutarray.append(btssample['image_path'])
        calcoutarray.append(calcdepth)
    print("Median:")
    eval(calcoutarray, gt)
    print("\n")
    return calcoutarray, pathoutarray

# Output an array to visualize
def eval_vis(x0,x1,x2):
    calcoutarray = []
    pathoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(btsoutarray, vnloutarray, sharpnetoutarray):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean3(btsdepth, vnldepth, sharpnetdepth, x0,x1,x2)
        pathoutarray.append(btssample['image_path'])
        calcoutarray.append(calcdepth)

    eval(calcoutarray)
    return calcoutarray, pathoutarray

# Combine the input models using a sigma clip
# Output the calculated metrics
def eval_sigmaclip():
    calcoutarray = []
    for btssample, vnlsample, sharpnetsample in zip(btsoutarray, vnloutarray, sharpnetoutarray):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        vnldepth = vnldepth * 10.0
        sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation

        sigmaclipdepth = sigma_clip([btsdepth, vnldepth, sharpnetdepth], sigma=1, axis=0)
        calcdepth['pred_depth'] = np.average(sigmaclipdepth, axis=0)
        calcdepth['path'] = btssample['path']
        calcoutarray.append(calcdepth)

    print("Done sigma calc")
    _,_,_,_,rms,_,_,_,_ = eval(calcoutarray)
    print(rms.mean())

# Create a visualization of the deptharray
def out_vis(outdir, firstname, deptharray, patharray):

    filenamesfile = "bts_config/bts_filenames_partial_train.txt"
    filenames = []

    with open(filenamesfile, 'r') as f:
        for line in f:
            imgname = line.split()[0].rsplit('/',1)[1]
            print(imgname)
            filenames.append(imgname)
    n = 0
    for sample, path in zip(deptharray, patharray):
        path_imgname = path.rsplit('/',1)[-1]
        if path_imgname not in filenames:
            continue

        finpath = firstname + "_nyu_image_" + str(n) + ".png"
        outpath = outdir + finpath
        print(outdir)
        print(outpath)

        samplescaled = sample * 1000.0
        samplescaled = samplescaled.astype(np.uint16)

        samplescaled = samplescaled * 5.75

        samplescaled = (samplescaled * (255.0 / 65535.0)).astype(np.uint8);
        color_meanscaled = cv2.applyColorMap(samplescaled, cv2.COLORMAP_JET)

        cv2.imwrite(outpath, color_meanscaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        n += 1

# Split array into train and validate sets
def splitarray(array, cfg_dir):

    filenames_train = os.path.join(cfg_dir, "bts_filenames_full_test_train.txt")
    filenames_val = os.path.join(cfg_dir, "bts_filenames_full_test_val.txt")

    filetrain = []
    fileval = []

    with open(filenames_train, 'r') as f:
        for line in f:
            filetrain.append(line.split()[0])

    outarray_train = []
    outarray_val = []

    for sample in array:
        if sample['image_path'] in filetrain:
            outarray_train.append(sample)
        else:
            outarray_val.append(sample)

    return outarray_train, outarray_val

# Split array into train and validate sets
def splitarray_white(array, cfg_dir):

    filenames_train = os.path.join(cfg_dir, "bts_filenames_full_test_train.txt")
    filenames_val = os.path.join(cfg_dir, "bts_filenames_full_test_val.txt")

    filetrain = []
    fileval = []

    with open(filenames_train, 'r') as f:
        for line in f:
            filetrain.append(line.split()[0])

    outarray_train = []
    outarray_val = []

    for sample in array:
        name = sample['image_path']
        name = name.replace("W.jpg", ".jpg")
        if name in filetrain:
            outarray_train.append(sample)
        else:
            outarray_val.append(sample)

    return outarray_train, outarray_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fuse NYU load')

    parser.add_argument('--cfg_dir', type=str, help='Path to config dir', default='./bts_config/')
    parser.add_argument('--load_dir_path', type=str, help='Path to the dir where the binary files to load are stored', default='./data_save/')
    parser.add_argument('--save_name', type=str, help='The additional save name provided in the run script. Leave empty if not used.', default='')
    parser.add_argument('--run_optimizations', help='Run all weighted average optimizations', action='store_true')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset dir', default='./datasets/nyu/')

    args = parser.parse_args()

    if (args.save_name == ''):
        btsoutarray = np.load(args.load_dir_path + 'nyu_bts.npy', allow_pickle=True)
        vnloutarray = np.load(args.load_dir_path + 'nyu_vnl.npy', allow_pickle=True)
        sharpnetoutarray = np.load(args.load_dir_path + 'nyu_sharpnet.npy', allow_pickle=True)
    else:
        btsoutarray = np.load(args.load_dir_path + 'nyu_bts_' + args.save_name + '.npy', allow_pickle=True)
        vnloutarray = np.load(args.load_dir_path + 'nyu_vnl_' + args.save_name + '.npy', allow_pickle=True)
        sharpnetoutarray = np.load(args.load_dir_path + 'nyu_sharpnet_' + args.save_name + '.npy', allow_pickle=True)

    # Split arrays into train and val
    btsoutarray_train, btsoutarray_val = splitarray(btsoutarray, args.cfg_dir)
    vnloutarray_train, vnloutarray_val = splitarray(vnloutarray, args.cfg_dir)
    sharpnetoutarray_train, sharpnetoutarray_val = splitarray_white(sharpnetoutarray, args.cfg_dir)

    gt_depths, gt_depths_train, gt_depths_val = loadgt(btsoutarray, args.cfg_dir, args.dataset_path)

    # ==========================
    # Table 1
    # ==========================
    if True:
        x0 = np.array([1.0,1.0,1.0])

        # Base methods
        print_array(x0, [0,1,0], btsoutarray, vnloutarray, sharpnetoutarray, gt_depths)
        print_array(x0, [0,0,1], btsoutarray, vnloutarray, sharpnetoutarray, gt_depths)
        print_array(x0, [1,0,0], btsoutarray, vnloutarray, sharpnetoutarray, gt_depths)

        # Median
        eval_median(btsoutarray, vnloutarray, sharpnetoutarray, gt_depths)

        # Fusion-avg
        print_array(x0, [1,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # Fusion-w-avg
        if args.run_optimizations:


            x0 = np.array([1.0,1.0,1.0])

            res = minimize(eval_array, x0, args=(btsoutarray_train, vnloutarray_train, sharpnetoutarray_train, gt_depths_train), method='nelder-mead',
                   options={'xatol': 1e-2, 'disp': True})

            print_array(res['x'], [1,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        else:
            x0 = np.array([0.9620615072833809,  0.8309236423351789, 1.2018215719295504])
            print_array(x0, [1,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

    # ==========================
    # Table 2
    # ==========================
    if True:
        # BTS x VNL
        x0 = np.array([1.0,1.0,1.0])
        # avg
        print_array(x0, [1,1,0], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # w-avg

        if args.run_optimizations:

            x0 = np.array([1.0,1.0])

            res = minimize(eval_two_array, x0, args=([1,1,0], btsoutarray_train, vnloutarray_train, sharpnetoutarray_train, gt_depths_train), method='nelder-mead',
                   options={'xatol': 1e-2, 'disp': False})

            print_array(res['x'], [1,1,0], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        else:
            x0 = np.array([1.0845581054687499,  0.9005615234374998, 0.0])
            print_array(x0, [1,1,0], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # VNL x SN
        x0 = np.array([1.0,1.0,1.0])
        # avg
        print_array(x0, [0,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # w-avg

        if args.run_optimizations:

            x0 = np.array([1.0,1.0])

            res = minimize(eval_two_array, x0, args=([0,1,1], btsoutarray_train, vnloutarray_train, sharpnetoutarray_train, gt_depths_train), method='nelder-mead',
                   options={'xatol': 1e-2, 'disp': False})

            print_array(res['x'], [0,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        else:
            x0 = np.array([0.0,  0.99951171875, 1.013427734375])
            print_array(x0, [0,1,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # BTS x SN
        x0 = np.array([1.0,1.0,1.0])
        # avg
        print_array(x0, [1,0,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        # w-avg
        if args.run_optimizations:

            x0 = np.array([1.0,1.0])

            res = minimize(eval_two_array, x0, args=([1,0,1], btsoutarray_train, vnloutarray_train, sharpnetoutarray_train, gt_depths_train), method='nelder-mead',
                   options={'xatol': 1e-2, 'disp': False})

            print_array(res['x'], [1,0,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)

        else:
            x0 = np.array([1.053424072265625, 0.0, 0.94737548828125])
            print_array(x0, [1,0,1], btsoutarray_val, vnloutarray_val, sharpnetoutarray_val, gt_depths_val)



    if False:
        eval_sigmaclip()

    if False:
        visarray, patharray = eval_vis(0.0,0.0,1.0)
        out_vis('./dataout/vis/TEST/sharpnet/', 'sharpnet_full_train', visarray, patharray)
