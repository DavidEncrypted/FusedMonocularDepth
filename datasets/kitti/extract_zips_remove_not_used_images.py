import os
import glob
from zipfile import ZipFile
import sys

basedir = "./data_depth_annotated/test"

filenames = []
with open("eigen_test_files_with_gt.txt", 'r') as f:
    for line in f:
        fullname = line.split()[0]
        #print(fullname)
        #fullname
        if fullname not in filenames:
            filenames.append(fullname)

file_name = "x"

ziplist = glob.glob('*.zip')
arialist = glob.glob('*.aria2')
for i in range(0,len(arialist)):
    arialist[i] = arialist[i][:-6]

# for file in arialist:
#     print(file)

# for file in ziplist:
#     print(file)
#
for file in ziplist:
    if file not in arialist:
        print("Processing file: ", file)
        with ZipFile(file, 'r') as zip:
            #print(zip.namelist())
            #continue
            for zipfile in zip.namelist():
                if zipfile in filenames:
                    #print(zipfile)
                    zip.extract(zipfile,basedir)
        print("Finished file: ", file)
# exit(0)
#
# with ZipFile(file_name, 'r') as zip:
#     #print(zip.namelist())
#     for zipfile in zip.namelist():
#         if zipfile in filenames:
#             print(zipfile)
#             zip.extract(zipfile,basedir)
