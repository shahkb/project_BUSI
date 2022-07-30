### Import packages
import os
import re
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from skimage import exposure


### Helper functions to build the data structure and extract masks corresponding to each image

# Initialize image file dictionary
#   file_name_fullpath, file_name, file_stem, file_suffix, file_masks, 
def init_img_dict():

    return dict( {'file_name_fullpath': None, 'file_name': None, 
                  'file_stem': None, 'file_suffix': None, 
                  'file_masks': None})


# Build a list of image file dictionary
# Given a directory, create a list of dictionary containing img_file names and corresponding masks
def get_file_dicts(img_dir):

    # all files in a directory (only files are returned, not directories)
    all_file_names = [file for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]

    # Split files into two categories - image and mask
    img_file_names = filter_files(all_file_names, isMask= False)
    mask_file_names = filter_files(all_file_names, isMask= True)

    assert len(img_file_names) + len(mask_file_names) == len(all_file_names), "Count of (mask + image) files is not equal to count of files"

    # Create data structure for holding image file names and their corresponding masks
    # Create a list of image file dictionary
    img_dict_list = list()
    for img_file in img_file_names:
        img_dict = init_img_dict()
        img_dict['file_name'] = img_file
        img_dict['file_name_fullpath'] = os.path.join(img_dir, img_file)
        img_dict['file_stem'] = Path(img_file).stem.split('.')[0]
        img_dict['file_suffix'] = Path(img_file).suffix
        mask_list = find_mask(img_file, mask_file_names)
        img_dict['file_masks'] = [os.path.join(img_dir, mask) for mask in mask_list]
        img_dict_list.append(img_dict)

    return img_dict_list


    # Filter file names with or without "_mask" in their names from a list of file names
def filter_files(file_list, isMask = True):

    # Pattern to identify files with "_mask" in their names
    #re_mask = r"\w+\s+\(\d+\)_mask*."
    re_mask = r"\w+\s+\(\d+\)(_vflip)*(_\d+)*_mask*."
    pattern_mask = re.compile(re_mask)
    if (isMask):
        files = [file for file in file_list if pattern_mask.search(file)]
    else:
        files = [file for file in file_list if pattern_mask.search(file) is None]

    return files


# Find mask file names corresponding to a image file name, e.g. 
#   'benign (1)_mask.png' should match benign (1).png
# There can be more than one mask per image file
# returns a list with file masks corresponding to the the file name
def find_mask(file_name, mask_file_name_list):

    file_stem = Path(file_name).stem.split('.')[0]
    file_suffix = Path(file_name).suffix
    # Pattern to extract mask for a given file 
    rstring = rf"{re.escape(file_stem)}_mask_*\d*{re.escape(file_suffix)}"
    pattern = re.compile(rstring) 

    file_masks = [file for file in mask_file_name_list if pattern.search(file)]

    return file_masks
    

def print_ndarray_info(ndarray, title):
    print(f"{title} [ndim, shape, dtype, min, max]: [{ndarray.ndim}, {ndarray.shape}, {ndarray.dtype}, {ndarray.min()}, { ndarray.max()}]")
    return


#############################################################################
### Image read, write, resize, fft, histogram_equalization and display images
#############################################################################

RESIZED_ROW, RESIZED_COL = 512, 512
FIGSIZE_1 = (6, 6)
FIGSIZE_3 = (16, 8)

# Read image and convert to floating i.e. 0.0 - 1.0
def img_read(img_file, toFloat = False):

    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if toFloat:
        img = img.astype(np.float32)
        if np.max(img) > 1:
            img /= 255.0

    return img

# most image formats require UINT so image has to be converted to UINT
def img_write(img, img_file):
    img_unit8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(img_file, img_unit8)


# Resize image to specific width and height
def img_resize(img, dim= (RESIZED_ROW, RESIZED_COL)):
# dim is (width, height)
    img_new = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img_new


# flip image about x axis, y axis, or both (0, 1 or -1)
def img_flip(img, flip_axis= 1):
    img_new = cv2.flip(img, flip_axis)
    return img_new

  
# Compute Fourier magnitude
def comp_fft(in_img):

    # fourier image
    [ydim, xdim] = in_img.shape
    win = np.outer(np.hanning(ydim), np.hanning(xdim))
    win = win/np.mean(win)

    # fourier image
    F = np.fft.fftshift(np.fft.fft2(in_img*win))
    Fmag = np.abs(F)
    Fmag[Fmag < 0.01] = 0.01

    return Fmag


# histogram equalization
def histogram_equalization(in_img):
  # compute cdf
  img_cdf, bins = exposure.cumulative_distribution(in_img, 256)
  
  # create empty array for all possible pixel values
  new_cdf = np.zeros(256)

  # populate array with values from cdf
  # use bins as the index into the array
  new_cdf[bins] = img_cdf

  # create empty array the same size as the image
  out_img = np.zeros(in_img.shape, dtype=in_img.dtype)

  # for each pixel, look up the value from the cdf
  for i in range(out_img.shape[0]):
    for j in range(out_img.shape[1]):
      out_img[i, j] = (new_cdf[ in_img[i, j] ] * 255)

  return out_img
    

# Display an image
def display_img(img, img_title):

    plt.figure(figsize = FIGSIZE_1)
    plt.title(img_title)
    plt.imshow(img, cmap='gray', vmin= np.min(img), vmax= np.max(img))
    plt.show()


# Display 3 images from a image list
def display_img_list_3(img_list, img_title, with_mask = False, with_hist = False, with_fft = False, log = False):

    img1 = img_read(img_list[0]['file_name_fullpath'], toFloat = True)
    img2 = img_read(img_list[1]['file_name_fullpath'], toFloat = True)
    img3 = img_read(img_list[2]['file_name_fullpath'], toFloat = True)
    display_3_imgs([img1, img2, img3], [img_title]*3)

    if with_mask:
        img4 = img_read(img_list[0]['file_masks'][0], toFloat = True)
        img5 = img_read(img_list[1]['file_masks'][0], toFloat = True)
        img6 = img_read(img_list[2]['file_masks'][0], toFloat = True)
        img_title_mask = img_title + " (Mask)"
        display_3_imgs([img4, img5, img6], [img_title_mask]*3)

    if with_hist:
        img_title_hist = "Histogram of " + img_title 
        display_3_hist([img1, img2, img3], [img_title_hist]*3)

    if with_fft:
        img_title_fft = "Fourier magnitude of " + img_title
        img1_Fmag = comp_fft(img1)
        img2_Fmag = comp_fft(img2)
        img3_Fmag = comp_fft(img3)
        if log:
            img1_Fmag = np.log(img1_Fmag)
            img2_Fmag = np.log(img2_Fmag)
            img3_Fmag = np.log(img3_Fmag)
        display_3_imgs([img1_Fmag, img2_Fmag, img3_Fmag], [img_title_fft]*3)

    return


# Display 3 images in a row
def display_3_imgs(img_list, img_title_list):

    img1, img2, img3 = img_list[0], img_list[1], img_list[2]
    img1_title, img2_title, img3_title = img_title_list[0], img_title_list[1], img_title_list[2]
    
    plt.figure(figsize = FIGSIZE_3)
    plt.subplot(1,3,1)
    plt.imshow(img1, cmap='gray', vmin= np.min(img1), vmax= np.max(img1))
    plt.title(img1_title)
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap='gray', vmin= np.min(img2), vmax= np.max(img2))
    plt.title(img2_title)
    plt.subplot(1,3,3)
    plt.imshow(img3, cmap='gray', vmin= np.min(img3), vmax= np.max(img3))
    plt.title(img3_title)
    plt.show()
    
    return


# Display histogram of 3 images in a row
def display_3_hist(img_list, img_title_list):

    img1, img2, img3 = img_list[0], img_list[1], img_list[2]
    img1_title, img2_title, img3_title = img_title_list[0], img_title_list[1], img_title_list[2]

    plt.figure(figsize = FIGSIZE_3)
    plt.subplot(1,3,1)
    plt.hist(img1.ravel(), bins=256, range=(np.min(img1), np.max(img1)))
    plt.title(img1_title)
    plt.subplot(1,3,2)
    plt.hist(img2.ravel(), bins=256, range=(np.min(img2), np.max(img2)))
    plt.title(img2_title)
    plt.subplot(1,3,3)
    plt.hist(img3.ravel(), bins=256, range=(np.min(img3), np.max(img3)))
    plt.title(img3_title)
    plt.show()

    return


# Resize all the images in a image list (of img_dict objects). 
# Write resized images to a (existing) directory. 
# Resized img file name has "_img_size" in its name.
# img_size is assumed to be same in both dimensions (i.e. aspect ratio = 1) 
def resize_imgs(img_list, img_dir_out, img_size):

    count = 0
    for img_dict in img_list:

        file_name_fullpath = img_dict['file_name_fullpath']
        file_stem = img_dict['file_stem']
        file_suffix = img_dict['file_suffix']
        file_masks = img_dict['file_masks']

        # file_masks should not be empty
        assert file_masks, f"Mask not found for image {file_name_fullpath}"
        file_mask = file_masks[0]

        img = img_read(file_name_fullpath, toFloat = True)
        img_mask = img_read(file_mask, toFloat = True)

        img_new = img_resize(img, dim= (img_size, img_size))
        img_mask_new = img_resize(img_mask, dim= (img_size, img_size))

        file_id = '_' + str(img_size)

        img_new_file_name = file_stem + file_id + file_suffix
        img_mask_new_file_name = file_stem + file_id + '_mask' + file_suffix

        img_write(img_new, os.path.join(img_dir_out, img_new_file_name))
        img_write(img_mask_new, os.path.join(img_dir_out, img_mask_new_file_name))
        count += 1

    print(f"Number of images resized to {img_size} resolution: {count}")


# Flip all the images in a image list (of img_dict objects). 
# Write the flipped images to a (existing) directory. 
# Flipped img file name has "_vflip" in its name.
# flip_axis= 0 is about x axis, 1 is about y axis, -1 is about xy
def flip_imgs(img_list, img_dir_out, flip_axis= 1):

    count = 0
    for img_dict in img_list:

        file_name_fullpath = img_dict['file_name_fullpath']
        file_stem = img_dict['file_stem']
        file_suffix = img_dict['file_suffix']
        file_masks = img_dict['file_masks']

        # file_masks should not be empty
        assert file_masks, f"Mask not found for image {file_name_fullpath}"
        file_mask = file_masks[0]

        img = img_read(file_name_fullpath, toFloat = True)
        img_mask = img_read(file_mask, toFloat = True)

        img_new = img_flip(img, flip_axis)
        img_mask_new = img_flip(img_mask, flip_axis)

        file_id = '_' + 'vflip'

        img_new_file_name = file_stem + file_id + file_suffix
        img_mask_new_file_name = file_stem + file_id + '_mask' + file_suffix

        img_write(img_new, os.path.join(img_dir_out, img_new_file_name))
        img_write(img_mask_new, os.path.join(img_dir_out, img_mask_new_file_name))
        count += 1
    
    print(f"Number of images flipped: {count}")


# Read image data from an image list and append the image data to to img_data
def append_img_data(img_data, img_list):

    for img_dict in img_list:

        file_name_fullpath = img_dict['file_name_fullpath']
        file_stem = img_dict['file_stem']
        file_suffix = img_dict['file_suffix']
        file_masks = img_dict['file_masks']
        
        # file_masks should not be empty
        assert file_masks, f"Mask not found for image {file_name_fullpath}"
        file_mask = file_masks[0]

        img = img_read(file_name_fullpath, toFloat = True)
        img_mask = img_read(file_mask, toFloat = True)

        img = img.reshape((1, -1))
        img_mask = img_mask.reshape((1, -1))

        img_data = np.append(img_data, img, axis=0)
    
    return img_data

