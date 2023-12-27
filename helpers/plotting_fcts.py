import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def combineImgs(
    bool_imgs:list,
    colors:list,
):
    """
    Combines a list of boolean images into one image
    and colors the pixels according to the colors list.
    Args:
        bool_imgs: list of boolean images; list of numpy arrays
        colors: list of colors; list of strings
    """
    rgb_img = np.zeros((bool_imgs[0].shape[0], bool_imgs[0].shape[1], 4), dtype=float)
    for i in range(len(bool_imgs)):
        rgb_img[bool_imgs[i]] = matplotlib.colors.to_rgba(colors[i])

    rgb_img = (255 * rgb_img).astype(np.uint8)
    return rgb_img