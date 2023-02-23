# Bachelors Thesis David Schep

This repo contains the code for my thesis for the Bachelor of Computer Science at the University of Leiden.
The file `thesis_final_v3.pdf` contains the thesis.


# CREDITS:
```
### The code in ./bts and much of the evaluation code used in the load scripts
https://github.com/cogaplex-bts/bts
### The code in ./vnl
https://github.com/YvanYin/VNL_Monocular_Depth_Prediction
### The code in ./sharpnet
https://github.com/MichaelRamamonjisoa/SharpNet
```

# Dependencies
Make sure these dependencies are installed:
```
pytorch
torchvision

numpy
imageio
pillow
matplotlib
opencv-python
dill
scipy
yaml
```
Pytorch version 1.4.0 was used to run these models.  

# Download models
## BTS
```
cd ./bts
mkdir models
cd models
wget https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip
wget https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet161.zip
unzip bts_nyu_v2_pytorch_densenet161.zip
unzip bts_eigen_v2_pytorch_densenet161.zip
```

## VNL
Download the model trained on NYU dataset [here](https://cloudstor.aarnet.edu.au/plus/s/7kdsKYchLdTi53p) 
Download the model trained on KITTI dataset [here](https://cloudstor.aarnet.edu.au/plus/s/eviO16z68cKbip5) 
Place the model files in `/vnl/`

## Sharpnet
Download the model  trained on NYU dataset [here](https://drive.google.com/open?id=1UTruzxPxQdoxF44X7D27f8rISFU0bKMK)
Place the model file in `/sharpnet/models/`

# Download datasets
## KITTI
To install the KITTI dataset download the ground truth data from this site: http://www.cvlibs.net/download.php?file=data_depth_annotated.zip.    
Extract it to `./datasets/kitti/data_depth_annotated`.  
``` bash
cd ./datasets/kitti
### Download all kitti archives containing images used as the test set images.
aria2c -x 16 -i ./kitti_only_test_archives.txt

### Extract from the zips only the used images according to `eigen_test_files_with_gt.txt`
python3 extract_zips_remove_not_used_images.py

```


## NYU Depthv2
``` bash
cd ../nyu

### Download the dataset
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

###
python3 extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat .

```

# Usage

The usage of the fused models is divided into four scripts. Two for each dataset. The first script runs the models on the dataset. This script is called `full_run_*.py`. It will run the models and then store the outputs in binary files. These files are each the same size as the entire dataset and are output for each model. This takes up quite some space but makes prototyping and usage a lot nicer as the models do not have to be run every time the fusion method is slightly changed.  
The `full_load_*.py` will load the stored outputs and fuse them together. It will then evaluate the different fusion method and output the evaluation metrics for each different fusion method. The outputs of these scripts are ordered to be identical to the order of the entries in the experiment tables in the thesis.  

## Run on NYU Depthv2

```
### Run from the root of the project
python3 full_run_nyu.py
python3 full_load_nyu.py
```
## Run on KITTI

```
### Run from the root of the project
python3 full_run_kitti.py
python3 full_load_kitti.py
```
## Dataset location
if the dataset is not located in `./datasets/nyu` or `./datasets/kitti`. The script can be pointed to the correct folder using the `--dataset_path` parameter. For nyu it needs to point to the directory containing the `test` folder. For KITTI it need to point at the `data_depth_annotated` folder.

## Running the optimizations
The load scripts have the capability to run the optimizations on the weights for the weighted averages. To turn this on use the `--run_optimizations` argmument.

## Other arguments
The scripts allow certain other arguments. Use `-h` to show them.


# TODO
- [ ] Add script to process a single image and output the depth
