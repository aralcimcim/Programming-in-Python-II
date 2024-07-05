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

Example solutions for tasks in the provided tasks file.
"""

import gzip

import dill as pkl
import numpy as np

#
# Task 1
#

# Write a function that converts a color image, represented as numpy array of
# shape (H, W, 3) into a grayscale image, represented as numpy array of shape
# (H, W). The color channels are in order RGB. The numpy array is of type
# np.uint8 with values from 0 to 255. The function should take 4 arguments: the
# numpy array representing the image and the 3 contributions of the red, green
# and blue channel (in this order). Default values for contributions of the red,
# green and blue channel should be r=0.2989, g=0.5870, b=0.1140 (this is just
# one way, check the link
# https://en.wikipedia.org/wiki/Grayscale#Converting_colour_to_grayscale
# for different ways of conversion). The function should return a numpy array of
# type np.uint8.
example_image = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)


def rgb2gray(rgb_array: np.ndarray, r=0.2989, g=0.5870, b=0.1140):
    grayscale_array = (rgb_array[..., 0] * r +
                       rgb_array[..., 1] * g +
                       rgb_array[..., 2] * b)
    grayscale_array = np.round(grayscale_array)
    grayscale_array = np.asarray(grayscale_array, dtype=np.uint8)
    return grayscale_array


#
# Task 2
#

# Find out programmatically which image folder belongs to the cluster as seen in
# the visualization of the RGB means and standard deviations of the main script.
# Hint: We know the cluster is somewhere around mean=[140, 150, 150] and
# std=[35, 40, 50] for channels R, G and B. We can use this to compute a
# distance from these three (mean, std) points to our data points. We can then
# sort the data points by that distance and locate the points close to these
# three points. If a lot of images from one folder are close, we have found our
# culprit! You can use np.argsort() to get data point indices sorted by distance
# and np.unique(..., return_counts=True) to get how often individual elements
# occur in an array.
with gzip.open("04_means_stds.pklz", "rb") as f:
    load_dict = pkl.load(f)
means = load_dict["means"]
stds = load_dict["stds"]
folder_names = load_dict["folder_names"]

# Compute some distance measure, e.g., Euclidean distance
distance_to_target = np.sqrt((means - [140, 150, 150]) ** 2 + (stds - [35, 40, 50]) ** 2)
# Get indices of data points, sorted by distance (in ascending order)
distance_to_target_inds = np.argsort(distance_to_target, axis=0)
# Get folder_names of, e.g., 100 closest data point indices
close_folder_names = folder_names[distance_to_target_inds[:100]]
# Check which folder name occurred a lot in the closest points
close_folder_names, counts = np.unique(close_folder_names, return_counts=True)
# Sort folder names by counts
sorted_inds = np.argsort(counts)
sorted_inds = sorted_inds[::-1]  # Indices of larger counts first
close_folder_names = close_folder_names[sorted_inds]
counts = counts[sorted_inds]
# Print folder names and counts
print(f"Close to cluster: {list(zip(close_folder_names, counts))}")
# Answer: Folder 009 is the cluster!


#
# Task 3
#

# Normalize a numpy array to range [0, 1].
# Normalize a numpy array to range [-1, 1].
array = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

array1 = (array - array.min()) / (array.max() - array.min())
array2 = 2 * array1 - 1  # [a, b] --> (b - a) * norm01data + a
print(f"array1: [{array1.min():.4f}, {array1.max():.4f}], mean={array1.mean():.4f}, std={array1.std():.4f}")
print(f"array2: [{array2.min():.4f}, {array2.max():.4f}], mean={array2.mean():.4f}, std={array2.std():.4f}")


#
# Task 4
#

# Convert an RGB image to a grayscale image using the function in task 1, then
# standardize it (the standardized array has mean = 0 and std = 1).
array = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

grayscale_array = rgb2gray(array)
standardized = (grayscale_array - grayscale_array.mean()) / grayscale_array.std()
print(f"min={standardized.min()}, max={standardized.max()}")
print(f"mean={standardized.mean()}, std={standardized.std()}")
