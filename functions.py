import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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


def sharpener(grey_img_array,colour_image_path):
    im_colour = Image.open(colour_image_path)
    colour_array = np.asarray(im_colour)
    grey_array = grey_img_array
    kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

    edge_matrix = convolve2d(grey_array, kernel, mode='same',
                               boundary='fill', fillvalue=0)
    edge_array = np.asarray(edge_matrix)

    edge_array_3d = edge_array[:, :, np.newaxis]
    k = 1.0
    im_sharpened_rgb = colour_array + (k * edge_array_3d)
    im_sharpened_rgb_clean = np.clip(im_sharpened_rgb, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(colour_array.astype(np.uint8))
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')

    axes[1].imshow(im_sharpened_rgb_clean)
    axes[1].set_title("Sharpened RGB Image")
    axes[1].axis('off')

    plt.show()

    return Image.fromarray(im_sharpened_rgb_clean)
