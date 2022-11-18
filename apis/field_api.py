import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects


def __get_field(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    ## mask of green (30,25,25) ~ (90, 255,255)
    mask = cv2.inRange(hsv, (30, 25, 25), (90, 255,255))

    # remove noise
    denoised_mask = remove_small_holes(~mask, area_threshold=30_000)
    denoised_mask = remove_small_objects(denoised_mask, min_size=30_000)

    ## slice the green
    imask = ~denoised_mask > 0
    field = np.zeros_like(frame, np.uint8)
    field[imask] = frame[imask]  # = 1 if you want the mask to train segmentation model
    
    return field


def __get_faster_field(frame, row_step=10, green_prop=0.5):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    ## mask of green (30,25,25) ~ (90, 255,255)
    mask = cv2.inRange(hsv, (30, 25, 25), (90, 255,255))

    # remove noise
    upper_row = 0
    lower_row = mask.shape[0]
    for row in range(0, mask.shape[0], row_step):
        green_pixels = np.count_nonzero(mask[row, :])
        if green_pixels > mask.shape[1] * green_prop:
            upper_row = row
            break
    
    for row in range(mask.shape[0]-1, 0, -row_step):
        green_pixels = np.count_nonzero(mask[row, :])
        if green_pixels > mask.shape[1] * green_prop:
            lower_row = row
            break
    
    field = np.zeros_like(frame, np.uint8)
    field[upper_row:lower_row, :, :] = frame[upper_row:lower_row, :, :]
    
    return field


def __get_noisy_field(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    ## mask of green (30,25,25) ~ (90, 255,255)
    mask = cv2.inRange(hsv, (30, 25, 25), (90, 255,255))
    
    field = cv2.bitwise_and(frame, frame, mask=mask)
    
    return field


def __get_lines_tophat(field, filter_size=(3, 3)):
    # Getting the kernel to be used in Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)
    input_image = cv2.cvtColor(field, cv2.COLOR_RGB2GRAY)
    tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)

    tophat_img[tophat_img < 10] = 0
    tophat_img[tophat_img != 0] = 255 

    return tophat_img


def detect_fields(frames):
    fields = []
    lines = []
    for frame in frames:
        field = __get_faster_field(frame)
        fields.append(field)
        lines.append(__get_lines_tophat(field))

    return np.array(fields), np.array(lines)
