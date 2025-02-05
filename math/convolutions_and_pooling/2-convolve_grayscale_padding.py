#!/usr/bin/env python3
"""  that performs a convolution on grayscale images with custom padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Arg:
    - images is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    - kernel is a numpy.ndarray with shape (kh, kw)
      containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    - padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0`s
    You are only allowed to use two for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    output = np.zeros(shape=(m, output_h, output_w))
    # Apply padding
    images_padded = np.pad(
        images,
        ((0, 0),
         (ph, ph),
         (pw, pw)),
        mode="constant"
        )

    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
