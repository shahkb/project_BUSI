# DATASCI 281: Computer Vision Project
Section 1 \
Summer 2022, Wed 4:00P-5:29P

Instructor:
 - Rachel Brown

## Project Title: Detection of cancer in breast ultrasound images (BUSI)
Link to the dataset: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>

## Team: 
 1. Sruthi Machina <smachina@berkeley.edu>
 2. Kishan B Shah <shahkb@berkeley.edu>


## Repo structure and content
The repo contains files for image analysis and classification to detect benign and malignant tumors in breast ultrasound image dataset (BUSI). 

### Folder `Dataset_BUSI_with_GT` 
The contains data imported from Kaggle: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>. 

- The subfolder `normal` contains BUSI that were diagnosed as normal (no tumor)

- The subfolder `benign` contains BUSI that were diagnosed as benign tumor

- The subfolder `malignant` contains BUSI that were diagnosed as malignant tumor. Note that the `malignant` subfolder contain twice as many images than the Kaggle dataset. The additional images were created by flipping the original images about the Y-axis in order to balance the dataset between the benign and the malignant cases. 

 - Number of images in the normal dataset is `133`, number of images in the benign dataset is `437`, and the number of images the malignant dataset is `420`. Note that the malignant case has twice as many images than the Kaggle dataset (`210`). Additional images in the malignant dataset were created by flipping images about the Y-axis.

 - The subfolder `normal_512`, `normal_256`, `normal_128` and `normal_64` contains normal BUSI that were resized to `512 x 512`, `256 x 256`, `128 x 128` and `64 x 64` resolutions, respectively.

 - The subfolder `benign_512`, `benign_256`, `benign_128` and `benign_64` contains benign BUSI that were resized to `512 x 512`, `256 x 256`, `128 x 128` and `64 x 64` resolutions, respectively.

 - The subfolder `malignant_512`, `malignant_256`, `malignant_128` and `malignant_64` contains malignant BUSI that were resized to `512 x 512`, `256 x 256`, `128 x 128` and `64 x 64` resolutions, respectively.


### Jupyter notebooks

- The notebook `image_analysis.ipynb` performs analysis of images in the BUSI dataset. It includes:
  - Visual analysis with iamges and their mask, histogram and frequency
  - Feature Analysis using
      - Prefilter (Historgram equalization)
      - Edge feature extraction (Prewitt, Sobel, Canny)
      - Corner detection (Harris and Shi-Tomasi)
      - Feature descriptors (SIFT, ORB)

 - The notebook `image_augmentation.ipynb` augments the malignant images by flipping images about the Y-axis.

 - The notebook `image_resize.ipynb` preforms resizing of images in the 3 datasets to `512 x 512`, `256 x 256`, `128 x 128` and `64 x 64` resolutions.

 - The notebook `image_pca_analysis.ipynb` preforms PCA analysis of the benign and malignant images to reduce the dimensions.

 - The notebook `image_classification.ipynb` preforms classification of images to detect benign and malignant tumors. It uses logistic regression, and bag of visual words, with and without PCA for dimensionality reduction

### Python files

- `utils.py` contains helper functions used in Jupyter notebooks for reading, transforming, writing, and analyzing BUSI


### More to follow ......



