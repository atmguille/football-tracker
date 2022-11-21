import os
import cv2
import numpy as np
import glob
import tqdm
import argparse

import sys
sys.path.append('./yolov7/')
sys.path.append('./mmpose/')
from apis import yolo_api, pose_api
sys.path.append('./apis/')
from apis import preprocessing_api


def filter_already_processed_videos(args):
    video_paths_original = glob.glob(f'{args.src_directory}/*.{args.video_format}')
    already_processed = glob.glob(f'{args.dst_directory}/*.mp4')
    already_processed = {path[len(f'{args.dst_directory}/'):-len('.mp4')] for path in already_processed}
    video_paths_to_process = [path for path in video_paths_original if path[len(f'{args.src_directory}/'):-len(f'.{args.video_format}')] not in already_processed]

    if len(video_paths_original) != len(video_paths_to_process):
        print(f'Found {len(video_paths_original) - len(video_paths_to_process)} out of {len(video_paths_original)} videos already processed')
    if len(video_paths_to_process) == 0:
        print('All videos already processed!')
        exit()

    return video_paths_to_process


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_directory', type=str, required=True, help='Directory containing videos to process')
    parser.add_argument('--dst_directory', type=str, required=True, help='Directory to save processed videos, in .avi format')
    parser.add_argument('--video_format', type=str, required=False, default='mp4', help='Format of videos to process')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--yolo_resolution', type=int, required=False, default=1280, help='Resolution to use for YOLO inference')
    parser.add_argument('--yolo_batch_size', type=int, required=False, default=100, help='Batch size of frames to use for YOLOv7 inference')
    parser.add_argument('--pose_batch_size', type=int, required=False, default=50, help='Batch size of frames to use for pose inference')
    parser.add_argument('--n_players_skeletons', type=int, required=False, default=-1, help='Number of players closest to the ball to compute skeletons for, -1 for all players. Less means faster inference')
    parser.add_argument('--yolo_checkpoint_path', type=str, required=False, default='./models_checkpoints/yolov7.pt', help='Path to YOLOv7 checkpoint')
    parser.add_argument('--pose_checkpoint_path', type=str, required=False, default='./models_checkpoints/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth', help='Path to pose checkpoint')
    parser.add_argument('--pose_config_path', type=str, required=False, default='./mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py', help='Path to pose config')
    parser.add_argument('--filter_processed_videos', action='store_true', help='Whether to filter videos already processed stored in dst_directory')
    parser.add_argument('--render_players', action='store_true', help='Whether to render players on top of the black background')
    parser.add_argument('--zoom_ball', action='store_true', help='Whether to zoom on the ball')
    parser.add_argument('--print_time', action='store_true', help='Whether to print time taken for each part of the preprocessing for each video')
    args = parser.parse_args()

    if not os.path.exists(args.dst_directory):
        os.mkdir(args.dst_directory)
    
    video_paths_to_process = filter_already_processed_videos(args) if args.filter_processed_videos else glob.glob(f'{args.src_directory}/*.{args.video_format}')
    print(f'Processing {len(video_paths_to_process)} videos ...')

    yolo_model, yolo_infer_size, yolo_infer_stride = yolo_api.get_model(weights_path=args.yolo_checkpoint_path, infer_size=args.yolo_resolution, device=args.device)
    pose_model, pose_dataset_info = pose_api.get_model(pose_config=args.pose_config_path, pose_checkpoint=args.pose_checkpoint_path, device=args.device)

    for video_name in tqdm.tqdm(video_paths_to_process):
        video_cap = cv2.VideoCapture(video_name)
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        original_frames = []

        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        original_frames = np.array(original_frames)

        preprocessed_frames = preprocessing_api.preprocess_frames(original_frames, yolo_model, yolo_infer_size, 
                                                yolo_infer_stride, args.yolo_batch_size, args.n_players_skeletons,
                                                pose_model, pose_dataset_info, args.pose_batch_size, 
                                                render_players=args.render_players, zoom_ball=args.zoom_ball, print_time=args.print_time)

        # Convert back to BGR (only needed when storing the video. If the video is kept in memory, it can be kept in RGB)
        final_frames = np.empty_like(preprocessed_frames)
        for i in range(len(preprocessed_frames)):
            final_frames[i] = cv2.cvtColor(preprocessed_frames[i], cv2.COLOR_RGB2BGR)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        dst_video_name = f"{args.dst_directory}/{video_name[len(f'{args.src_directory}/'):-len(f'.{args.video_format}')]}.mp4"
        out = cv2.VideoWriter(dst_video_name, fourcc, fps, final_frames[0].shape[:2][::-1])

        for frame in final_frames:
            out.write(frame)
