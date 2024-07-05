# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will look into how to normalize and analyze our dataset. You
will need to download our dataset (or have access to a folder with similar files
and structure). Start by working on a small version of your dataset.
"""

################################################################################
# Our dataset
################################################################################

# Our dataset consists of RGB(A) images stored as .jpg files. The files are
# organized in folders, each folder representing an individual data collection
# process (not really, but for demonstration purposes, assume this is the case).
# The images are part of the following two datasets:
# 1) https://www.kaggle.com/datasets/andrewmvd/pollen-grain-image-classification
# 2) https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
# A subset of both datasets was selected to compose our image dataset of 10
# folders, each with 100 images, i.e., 1000 images in total.

# Path to the dataset (unzip the "04_images.zip" archive)
input_path = "04_images"

#
# Let's start by taking a look at our data by reading in one file
#

import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# Get list of all specified files
image_files = sorted(glob.glob(os.path.join(input_path, "**", "*.jpg"), recursive=True))
# Check number of found files
print(f"Found {len(image_files)} image files")

# Read first image file
with Image.open(image_files[0]) as im:  # This returns a PIL image
    image = np.array(im)  # We can convert it to a numpy array
print("image data:")
print(f"mode: {im.mode}; shape: {image.shape}; min: {image.min()}; max: {image.max()}; dtype: {image.dtype}")

# We are dealing with image data, so each sample is high-dimensional and
# contains as many features as it has pixels. Pixel values range from 0 to 255
# in 3 color channels for RGB and in 1 channel for grayscale images. In case
# transparency is also stored, we have an additional channel (alpha).

#
# Check means and standard deviations of images
#

# We know how many images to expect, so we can already allocate numpy arrays to
# store mean and std values as float values and the folder_names of the folders.
# Since we want RGB images, we will collect the metrics for each channel,
# dropping the transparency information in case an image has an alpha channel.
means = np.zeros(shape=(len(image_files), 3))
stds = np.zeros(shape=(len(image_files), 3))
folder_names = []

# Loop through files, read them, and store mean, std and folder folder_name
for i, image_file in tqdm(enumerate(image_files), desc="Processing files", total=len(image_files)):
    with Image.open(image_file) as im:
        image = np.array(im)
    # Check that we have RGB(A) images and drop the (potential) alpha channel
    assert len(image.shape) == 3 and image.shape[2] >= 3, f"{image_file}: {image.shape}"
    if image.shape[2] == 4:
        image = image[:, :, :-1]
    # Perform metric computations along axes 0 and 1 (height and width)
    means[i] = image.mean(axis=(0, 1))
    stds[i] = image.std(axis=(0, 1))
    folder_names.append(os.path.basename(os.path.dirname(image_file)))
folder_names = np.array(folder_names)

# It's a good idea to create save-points if computation takes a while. Here, we
# save the means, stds and folder_names that we computed.
import dill as pkl
import gzip

# Save our data in a compressed pickle file
with gzip.open("04_means_stds.pklz", "wb") as f:
    pkl.dump(dict(means=means, stds=stds, folder_names=folder_names), file=f)

# Load precomputed data from the compressed pickle file
# with gzip.open("04_means_stds.pklz", "rb") as f:
#     load_dict = pkl.load(f)
# means = load_dict["means"]
# stds = load_dict["stds"]
# folder_names = load_dict["folder_names"]

# Now we want to visualize our data. We will use a pyplot 2D scatter plot.
from matplotlib import pyplot as plt

fig, axes = plt.subplots(ncols=3, figsize=(16, 5), sharex=True, sharey=True)
unique_folder_names, point_colors = np.unique(folder_names, return_inverse=True)
for i in range(3):
    scatter = axes[i].scatter(x=means[:, i], y=stds[:, i], c=point_colors, s=5, cmap="nipy_spectral")
    axes[i].set_xlabel("mean")
    axes[i].set_ylabel("standard deviation")
    axes[i].grid(True)
    axes[i].set_title(f"channel {i}")
fig.legend(scatter.legend_elements()[0], unique_folder_names, loc="upper center", ncol=16)
fig.savefig('mean_std.pdf')


################################################################################
# Downprojection and visualization of our data
################################################################################

# We want to check for clusters in our data. We want to use, e.g., t-SNE, but
# our raw image data is not suited for this method. Therefore, we will use a
# pretrained CNN to downproject the images into a better suited feature space
# before we apply t-SNE.

#
# Projecting our images to a CNN feature space
#

# PyTorch gives us convenient access to datasets and pretrained models via
# torchvision. The original models and data might not be hosted by PyTorch
# itself, so they might change, and we thus should always keep a local copy if
# we want reproducibility.
import torch
import torchvision

# Use CUDA device (GPU) if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# We can choose from a variety of models:
# https://pytorch.org/vision/stable/models.html
# We want some model that has few features before its output layer and was
# trained on targets that are related to our task.

# Get SqueezeNet 1.1 model, pretrained on ImageNet
weights = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
pretrained_model = torchvision.models.squeezenet1_1(weights=weights)
# Send pretrained_model to GPU (or keep on CPU if device="cpu")
pretrained_model = pretrained_model.to(device=device)
# Switch to evaluation mode to ensure we are using the correct model behavior
pretrained_model.eval()
# Get required data preprocessing transformations that must be applied before we
# forward our image data to the pretrained model (what exactly must be done here
# depends on how the model was originally trained; luckily, the .transforms()
# method here can be used to obtain all the required transformations)
preprocess = weights.transforms()

# Again, we can allocate the numpy array beforehand: We know how many images to
# expect, and we can look up that the layer before the output layer has 512
# features in SqueezeNet 1.1.
n_model_features = 512
images_projected_cnn = np.zeros(shape=(len(image_files), n_model_features), dtype=np.float32)

# Loop through image files and extract CNN features (we will later learn how to
# do this using optimized PyTorch tools)
with torch.no_grad():  # We do not need to store gradients for this
    for i, image_file in tqdm(enumerate(image_files), desc="Processing files", total=len(image_files)):
        # Open image file, convert to numpy array and then to torch tensor.
        # There is a utility function directly in torchvision.io (read_image),
        # but it does not support some images (yet).
        with Image.open(image_file) as im:
            image = torch.from_numpy(np.asarray(im, dtype=np.float32))
        # Drop the (potential) alpha channel
        if image.shape[2] == 4:
            image = image[:, :, :-1]
        # The original image dimension is (H, W, 3), but the pretrained model
        # expects (3, H, W). Also, the data is expected to be in range [0, 1].
        image = torch.movedim(image, 2, 0) / 255
        # Now, we can apply the required preprocessing.
        image = preprocess(image).to(device=device)
        
        # Now we can apply the pretrained model. The input to .forward() is
        # actually a tensor that can contain multiple samples (mini-batches), so
        # we need yet another dimension: (N, 3, H, W), where N is the number of
        # samples. In our case, N=1, i.e., we can simply add a new dimension
        # (either via slicing syntax or torch.unsqueeze())
        image_features = pretrained_model.features(image.unsqueeze(0))
        # Output is of shape (1, 512, H', W'), so we compute the mean over the
        # H' and W' dimensions (to ultimately obtain 512 features)
        image_features = torch.mean(image_features, dim=(2, 3))
        # Finally, we can store the 512 computed CNN features
        images_projected_cnn[i, :] = image_features.cpu()

#
# Applying t-SNE (adapt the code if you want to use a different algorithm)
#

# sklearn provides us with a nice t-SNE implementation and good documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
from sklearn.manifold import TSNE

# Get t-SNE model and fit (=train) it on our data
tsne = TSNE(n_components=2, perplexity=30, random_state=1)
images_projected_tsne = tsne.fit_transform(images_projected_cnn)
print(f"t-SNE projected our data to shape {images_projected_tsne.shape}")

# Save our results in case we want to use them later
np.savez_compressed("04_clustering", images_projected_tsne=images_projected_tsne,
                    images_projected_cnn=images_projected_cnn,
                    folder_names=folder_names, file_names=image_files)

# Load precomputed data from the compressed numpy file
# loaded = np.load("04_clustering.npz")
# images_projected_tsne = loaded["images_projected_tsne"]
# images_projected_cnn = loaded["images_projected_cnn"]
# folder_names = loaded["folder_names"]
# file_names = loaded["file_names"]

# Plot the result
fig, ax = plt.subplots()
scatter = ax.scatter(x=images_projected_tsne[:, 0], y=images_projected_tsne[:, 1],
                     c=point_colors, s=5, cmap="nipy_spectral")
ax.set_xticks([])
ax.set_yticks([])
fig.legend(scatter.legend_elements()[0], unique_folder_names, loc="upper center", ncol=5)
fig.savefig("tsne.pdf")

#
# Inspecting the downprojected images
#

# t-SNE is good for visualization, but we cannot blindly trust the results. To
# inspect our clusters and check what they mean/if they make sense, we have to
# take a look at the clustered images. This can give you important information
# about the properties of your dataset.
