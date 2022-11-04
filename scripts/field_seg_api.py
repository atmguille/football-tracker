from cgitb import small
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import cv2

def __expand_above(masks, lookup_step=2, expand_range=40):

    width = masks.shape[2]
    height = masks.shape[1]
    
    for (k, image) in enumerate(masks):
        point = [0,width//2]
        while point[0] < height and image[point[0], point[1]] == 0:
            point[0] += lookup_step
        point[0] += expand_range
        if point[0] >= height:
            continue
        
        v_pos = point[0]
        left = point[1]

        while left >= 0 and image[v_pos, left] == 1:
            left -= lookup_step
        
        if left < 0:
            left = 0

        right = point[1]

        while right < width and image[v_pos, right] == 1:
            right += lookup_step
        
        if right >= width:
            right = width    

        image[v_pos-2*expand_range:v_pos,left:right] = 1
        masks[k] = image
    
    return masks

def get_model(weights_path='../input/bundesliga-models/segmentation_field_best.h5', device='cuda:0'):

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_visible_devices(gpus[int(device[-1])], 'GPU')

    with tf.device('/gpu:' + device[-1]):
        model = keras.models.load_model(weights_path)

    return model

def detect(model, frames: np.ndarray, device='cuda:0', resize=False, expand_above=True, lookup_step=2, expand_range=40):
    """
    Set resize to True if different than 640 x 640
    """
    with tf.device('/gpu:' + device[-1]):

        original_size = frames.shape[1:3]

        if resize:
            # Hay que hacer resize
            resized_frames = np.empty((frames.shape[0], 640, 640, frames.shape[3]))
            for (k, image) in enumerate(frames):
                resized_frames[k] = cv2.resize(image, (640,640))
        else:
            resized_frames = frames

        # Ahora aplicamos el modelo 
        small_masks = np.argmax(model.predict(resized_frames), axis=-1).astype('uint8')

        if resize:
            # Ultimo resize
            masks = np.empty(frames.shape[:3])
            for (k, image) in enumerate(small_masks):
                masks[k] = cv2.resize(image, (original_size[1], original_size[0]))
        else:
            masks = small_masks

        if expand_above:
            masks = __expand_above(masks, lookup_step, expand_range)

        masks = masks.astype('uint8')

        return masks
