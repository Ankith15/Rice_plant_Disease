import numpy as np 
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
import shutil

# from keras.preprocessing.image import ImageDataGenerator

# Path to the dataset directory
datadir = "F:\\Rice_leaf_Disease\\leaf"

# Valid image extensions
img_ext = ('jpg', 'png', 'jpeg', 'bmp')  # Tuple for valid image extensions

# Dictionary to hold class-wise image paths
data = {}
for class_name in os.listdir(datadir):
    class_path = os.path.join(datadir, class_name)
    if os.path.isdir(class_path):
        data[class_name] = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(img_ext)]

# Check and split the dataset into train, validation, and test
from sklearn.model_selection import train_test_split

# Initialize dictionaries to hold the splits
train_data, val_data, test_data = {}, {}, {}

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split the data for each class
for class_name, images in data.items():
    train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    
    train_data[class_name] = train_imgs
    val_data[class_name] = val_imgs
    test_data[class_name] = test_imgs

# Print the number of images in each split
for class_name in data.keys():
    print(f"{class_name} - Train: {len(train_data[class_name])}, Validation: {len(val_data[class_name])}, Test: {len(test_data[class_name])}")


output_dir = 'split_leaf'
os.makedirs(output_dir,exist_ok  = True)

for split, split_data in zip(['train','validation','test'],[train_data,val_data,test_data]):
    for class_name, images in split_data.items():
        split_class_dir = os.path.join(output_dir,split,class_name)
        os.makedirs(split_class_dir, exist_ok = True)

        for img in images:
            shutil.copy(img,os.path.join(split_class_dir, os.path.basename(img)))

print('files have been split and saved')