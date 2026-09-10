import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def greyscaler(image_path):
    im = Image.open(image_path)

    im_array = np.asarray(im)
    # (Y=0.299 R+0.587G+0.114B )

    im_greyed = 0.299 * im_array[:, :, 0] + 0.587 * im_array[:, :, 1] + 0.114 * \
                im_array[:, :, 2]

    array_min, array_max = im_greyed.min(), im_greyed.max()

    if array_max > array_min:
        normalized_grey = (im_greyed - array_min) / (array_max - array_min) * 255
    else:
        normalized_grey = im_greyed * 0

    grey_image_array = normalized_grey.astype(np.uint8)

    return grey_image_array

def convcomponents(imagematrix,kernel_mat,i,j):
    # my own inefficient convolution function
    # I wanted to write it myself to directly apply my coursework.

    a = imagematrix.shape[0]
    b = imagematrix.shape[1]
    n = len(kernel_mat)
    d = (n)//2
    componentsum = 0

    if i == 0 and j ==0:
        for k in range(1, n):
            for l in range(1, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif i == a-1 and j == b-1:
        for k in range(0, n-1):
            for l in range(0, n-1):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif i == 0 and j == b-1:
        for k in range(1, n):
            for l in range(0, n-1):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif i == a-1 and j == 0:
        for k in range(0, n-1):
            for l in range(1, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif i == 0:
        for k in range(1,n):
            for l in range(0, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif j == 0:
        for k in range(0, n):
            for l in range(1, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif i == a-1:
        for k in range(0, n-1):
            for l in range(0, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    elif j == b-1:
        for k in range(0, n):
            for l in range(0, n-1):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])
    else:
        for k in range(0, n):
            for l in range(0, n):
                componentsum += (kernel_mat[k][l]) * (imagematrix[i - d + k][j - d + l])

    return componentsum

def sharpener(grey_img_array,colour_image_path):
    im_colour = Image.open(colour_image_path)
    colour_array = np.asarray(im_colour)
    grey_array = grey_img_array
    kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])
    dimensions = np.shape(grey_array)

    edge_array = np.zeros(shape=(dimensions[0], dimensions[1]))

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            edge_array[i][j] = convcomponents(grey_array, kernel, i, j)


    edge_array_3d = edge_array[:, :, np.newaxis]
    k = 1.0
    im_sharpened_rgb = colour_array + (k * edge_array_3d)
    im_sharpened_rgb_clean = np.clip(im_sharpened_rgb, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colour_image = Image.fromarray(im_sharpened_rgb_clean)
    colour_image.save("sharpened_cloud.png")
    axes[0].imshow(colour_array.astype(np.uint8))
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')

    axes[1].imshow(im_sharpened_rgb_clean)
    axes[1].set_title("Sharpened RGB Image")
    axes[1].axis('off')

    plt.show()

    return Image.fromarray(im_sharpened_rgb_clean)
