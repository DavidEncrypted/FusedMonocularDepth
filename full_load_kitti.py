import cv2
import numpy as np
import sys
from scipy.optimize import minimize
import argparse
import os

eigen_crop = False
do_kb_crop = True
garg_crop = True
min_depth_eval = 1e-3
max_depth_eval = 80
dataset = "kitti"


def calcmean(array1, array2, i, j):
    return (((array1 * i) + (array2 * j))/ (i+j))

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def compute_rmse(gt, pred):

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def eval_rmse(pred_depths, gt, missing_i_samples):
    num_samples = len(pred_depths)

    pred_depths_valid = []


    for t_id in range(num_samples):
        if t_id in missing_i_samples:
            continue

        pred_depths_valid.append(pred_depths[t_id])
    num_samples = num_samples - len(missing_i_samples)
    #print("removed: ", len(pred_depths) - len(pred_depths_valid), " samples")

    pred_depths = pred_depths_valid
    rms = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        if do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        if garg_crop or eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        rms[i] = compute_rmse(gt_depth[valid_mask], pred_depth[valid_mask])
    return rms

def eval(pred_depths, gt, missing_i_samples):

    num_samples = len(pred_depths)
    pred_depths_valid = []


    for t_id in range(num_samples):
        if t_id in missing_i_samples:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_i_samples)
    print("removed: ", len(pred_depths) - len(pred_depths_valid), " samples")
    pred_depths = pred_depths_valid

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)

    i = 0
    for i in range(num_samples):


        gt_depth = gt[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        # if args.do_kb_crop:
        #     height, width = gt_depth.shape
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        #     pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
        #     pred_depth = pred_depth_uncropped

        if do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        if garg_crop or eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3

def loadgt(outarray, config_dir):
    #global missing_i_samples
    missing_i_samples = []

    #global missing_i_samples_train
    missing_i_samples_train = []

    #global missing_i_samples_val
    missing_i_samples_val = []

    filenames_train = config_dir + "kitti_filenames_full_test_train.txt"
    filetrain = []
    with open(filenames_train, 'r') as f:
        for line in f:
            filetrain.append(line.split()[0])

    outarray_train = []
    outarray_val = []


    gt_depths = []

    i_train = 0
    i_val = 0

    for i_sample in range(len(outarray)):

        path = outarray[i_sample]['image_path']
        # TODO
        gt_depth_path = "/media/david/LinuxStorage/dataset/data_depth_annotated/gt/" + path.rsplit('/',4)[1] + "/proj_depth/groundtruth/image_02/" + path.rsplit('/',1)[1][:-4] + ".png"

        name = '/'.join(path.split('/')[-5:])

        if name[0] == '.':
            name = name[2:]
        # file_dir = pred_filenames[t_id].split('.')[0]
        # filename = file_dir.split('_')[-1]
        # directory = file_dir.replace('_rgb_'+file_dir.split('_')[-1], '')
        # gt_depth_path = os.path.join(args.gt_path, directory, 'sync_depth_' + filename + '.png')

        #
        depth = cv2.imread(gt_depth_path, -1)
        if depth is None:
            #print('Missing: %s ' % gt_depth_path)
            missing_i_samples.append(i_sample)

            if name in filetrain:
                missing_i_samples_train.append(i_train)
                i_train += 1
            else:
                missing_i_samples_val.append(i_val)
                i_val += 1

            continue

        depth = depth.astype(np.float32) / 256.0


        if name in filetrain:
            outarray_train.append(depth)
            i_train += 1
        else:
            outarray_val.append(depth)
            i_val += 1
        gt_depths.append(depth)

    print("num gt: ", len(gt_depths))
    print("num GT missing: ", len(missing_i_samples))
    print("gt train: ", len(outarray_train))
    print("gt val: ", len(outarray_val))
    print("i_train", i_train)
    print("i_val", i_val)
    return gt_depths, outarray_train, outarray_val, missing_i_samples, missing_i_samples_val, missing_i_samples_train

def print_array(y, bts, vnl, gt, mis_sam):
    calcoutarray = []
    for btssample, vnlsample in zip(bts, vnl):


        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        vnldepth = vnldepth * 80.0
        #sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean(btsdepth, vnldepth, y[0], y[1])
        calcoutarray.append(calcdepth)

    _,_,_,_,rms,_,_,_,_ = eval(calcoutarray, gt, mis_sam)
    #rms = eval(calcoutarray)
    print(y[0], " * BTS, ", y[1], " * VNL | RMSE: ", rms.mean())

def eval_array(y, bts, vnl, gt, mis_sam):

    calcoutarray = []
    for btssample, vnlsample in zip(bts, vnl):
        btsdepth = btssample['pred_depth']
        vnldepth = vnlsample['pred_depth']
        vnldepth = vnldepth * 80.0
        #sharpnetdepth = sharpnetsample['pred_depth']
        # do calculation
        calcdepth = calcmean(btsdepth, vnldepth, y[0], y[1])
        calcoutarray.append(calcdepth)

    rms = eval_rmse(calcoutarray, gt, mis_sam)
    print(y[0], y[1], rms.mean())
    return rms.mean()


def splitarray(array, config_dir):

    filenames_train = config_dir + "kitti_filenames_full_test_train.txt"
    filenames_val = config_dir + "kitti_filenames_full_test_val.txt"

    filetrain = []
    fileval = []

    with open(filenames_train, 'r') as f:
        for line in f:
            filetrain.append(line.split()[0])
            #print("f: ", line.split()[0])
    outarray_train = []
    outarray_val = []

    for sample in array:
        #print("k: ", sample['image_path'])
        normpath = os.path.normpath(sample['image_path'])
        name = '/'.join(normpath.split('/')[-5:])
        #print("s: ", splitar)
        #name = splitar[1] + '/' + splitar[2]
        #print(name)
        if name[0] == '.':
            name = name[2:]
        # print(name)
        if name in filetrain:
            outarray_train.append(sample)
        else:
            outarray_val.append(sample)

    print("train: ", len(outarray_train))
    print("val: ", len(outarray_val))
    return outarray_train, outarray_val


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fuse KITTI load')

    parser.add_argument('--cfg_dir', type=str, help='Path to config dir', default='./bts_config/')
    parser.add_argument('--load_dir_path', type=str, help='Path to the dir where the binary files to load are stored', default='./data_save/')
    parser.add_argument('--save_name', type=str, help='The additional save name provided in the run script. Leave empty if not used.', default='')
    parser.add_argument('--run_optimizations', help='Run all weighted average optimizations', action='store_true')


    args = parser.parse_args()


    if (args.save_name == ''):
        print("loading: ", args.load_dir_path + 'kitti_bts.npy')
        btsoutarray = np.load(args.load_dir_path + 'kitti_bts.npy', allow_pickle=True)
        print("loading: ", args.load_dir_path + 'kitti_vnl.npy')
        vnloutarray = np.load(args.load_dir_path + 'kitti_vnl.npy', allow_pickle=True)
    else:
        print("loading: ", args.load_dir_path + 'kitti_bts_' + args.save_name + '.npy')
        btsoutarray = np.load(args.load_dir_path + 'kitti_bts_' + args.save_name + '.npy', allow_pickle=True)
        print("loading: ", args.load_dir_path + 'kitti_vnl_' + args.save_name + '.npy')
        vnloutarray = np.load(args.load_dir_path + 'kitti_vnl_' + args.save_name + '.npy', allow_pickle=True)


    btsoutarray_train, btsoutarray_val = splitarray(btsoutarray, args.cfg_dir)
    vnloutarray_train, vnloutarray_val = splitarray(vnloutarray, args.cfg_dir)


    gt_depths, gt_depths_train, gt_depths_val, missing_i_samples, missing_i_samples_val, missing_i_samples_train = loadgt(btsoutarray, args.cfg_dir)



    # ==========================
    # Table 3
    # ==========================
    print_array([0.0,1.0], btsoutarray, vnloutarray, gt_depths, missing_i_samples)

    print_array([1.0,0.0], btsoutarray, vnloutarray, gt_depths, missing_i_samples)

    print_array([1.0,1.0], btsoutarray_val, vnloutarray_val, gt_depths_val, missing_i_samples_val)


    if args.run_optimizations:
        x0 = np.array([1.0,1.0])

        res = minimize(eval_array, x0, args=(btsoutarray_train, vnloutarray_train, gt_depths_train, missing_i_samples_train), method='nelder-mead',
               options={'xatol': 1e-2, 'disp': True})

        print_array(res['x'], btsoutarray_val, vnloutarray_val, gt_depths_val, missing_i_samples_val)
    else:
        print_array([1.552671813964844 ,  0.41797485351562336], btsoutarray_val, vnloutarray_val, gt_depths_val, missing_i_samples_val)
