from pickle import TRUE
import sys
import cv2
import numpy as np
import glob

sys.path.append('./yolov7/')
sys.path.append('./our_yolo/')
sys.path.append('./mmpose/')

import yolo_api
import our_yolo_api
import pose_api
import preprocessing_api
import field_seg_api

SRC_DIRECTORY = '../input/training_clips/original_short'
DST_DIRECTORY = '../input/training_clips/preprocessed_short_with_players_zoom'

LIST_VIDEOS_PATHS = glob.glob(f'{SRC_DIRECTORY}/*.avi')
TARGET_FPS = 'FULL'  # FULL or HALF
FRAMES_READ_BATCH_SIZE = 1000
DISTRIBUTE = False
DEVICE = 'cuda:0'

FIELD_SEG_WITH_MODEL = False

FIELD_BATCH_SIZE = 100
YOLO_RESOLUTION = 1280
YOLO_BATCH_SIZE = 150
POSE_BATCH_SIZE = 50
N_PLAYERS_SKELETONS = 5
UPDATE_RATE_CENTROIDS = 400
WEIGHTS_PATH_FIELD_SEG = '../input/bundesliga-models/segmentation_field_best.h5'
WEIGHTS_PATH_YOLO = './yolov7/yolov7.pt'
POSE_CONFIG_PATH = './mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py'
POSE_CHECKPOINT_PATH = './mmpose/checkpoints/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth'


if __name__ == '__main__':

    print('Loading models...')

    # Load models
    if FIELD_SEG_WITH_MODEL:
        field_seg_model = field_seg_api.get_model(WEIGHTS_PATH_FIELD_SEG, DEVICE)
    else:
        field_seg_model = None


    yolo_infer_size, yolo_infer_stride = None, None
    yolo_model, yolo_infer_size, yolo_infer_stride = yolo_api.get_model(weights_path=WEIGHTS_PATH_YOLO, infer_size=YOLO_RESOLUTION, device=DEVICE)

    pose_model, pose_dataset_info = pose_api.get_model(pose_config=POSE_CONFIG_PATH, pose_checkpoint=POSE_CHECKPOINT_PATH, device=DEVICE)

    # Filter videos already processed
    already_processed = glob.glob(f'{DST_DIRECTORY}/*.avi')
    already_processed = {path[len(f'{DST_DIRECTORY}/'):-len('.avi')] for path in already_processed}
    list_videos_paths = [path for path in LIST_VIDEOS_PATHS if path[len(f'{SRC_DIRECTORY}/'):-len('.avi')] not in already_processed]

    if DISTRIBUTE:
        list_videos_paths = sorted(list_videos_paths)
        if DEVICE == 'cuda:0':
            list_videos_paths = list_videos_paths[::2]
        elif DEVICE == 'cuda:1':
            list_videos_paths = list_videos_paths[1::2]

    for video_name in list_videos_paths:
        video_cap = cv2.VideoCapture(video_name)
        num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_read = 0

        while video_cap.isOpened() and frames_read < num_frames:
            original_frames = []

            frames_to_read = min(FRAMES_READ_BATCH_SIZE, num_frames - frames_read)

            for _ in range(frames_to_read):
                ret, frame = video_cap.read()
                if not ret:
                    break
                original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frames_read += frames_to_read

            original_frames = np.array(original_frames)

            preprocessed_frames = preprocessing_api.preprocess_frames(original_frames, field_seg_model, FIELD_BATCH_SIZE, yolo_model, yolo_infer_size, 
                                                    yolo_infer_stride, YOLO_BATCH_SIZE, N_PLAYERS_SKELETONS,
                                                    pose_model, pose_dataset_info, POSE_BATCH_SIZE, UPDATE_RATE_CENTROIDS, 
                                                    render_players=False, zoom_ball=False, print_time=False)

            # Convert back to BGR (only needed when storing the video! If the video is kept in memory, it can be kept in RGB)
            final_frames = np.empty_like(preprocessed_frames)
            for i in range(len(preprocessed_frames)):
                final_frames[i] = cv2.cvtColor(preprocessed_frames[i], cv2.COLOR_RGB2BGR)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            dst_video_name = f'{DST_DIRECTORY}/{video_name[len(f"{SRC_DIRECTORY}/"):-len(".avi")]}.avi'
            print(dst_video_name)
            out = cv2.VideoWriter(dst_video_name, fourcc, 25 if TARGET_FPS == 'FULL' else 12.5, final_frames[0].shape[:2][::-1])

            for frame in final_frames:
                out.write(frame)
