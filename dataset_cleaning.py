"""
Script to move the training files from dataset:

https://www.kaggle.com/gpiosenka/100-bird-species

into expected structure for PyTorch DCGAN tutorial
"""

import os
import shutil

src_dir = os.getcwd() + "/train"
dest_file = src_dir + "/dataset"

# for each folder:
for folder in os.listdir(src_dir):
    prepped_folder = folder.lower().replace(" ", "_")
    # for each file in the folder:
    data_paths = [os.path.join(pth, f) for pth, dirs, files in os.walk(src_dir) for f in files]
    print(data_paths)

    for file in os.listdir(f"{src_dir}/{folder}"):
        # rename it to have the bird name + number
        new_filename = f"{prepped_folder}_{file}"
        src = f"{src_dir}/{folder}/{file}"
        dest = f"./dataset/{new_filename}"
        # move it to the expected format
        shutil.copyfile(src,dest)
