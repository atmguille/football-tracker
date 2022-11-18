import time
import numpy as np
import torch

import field_api
import yolo_api
import players_ball_api
import pose_api
import drawing_api


def preprocess_frames(original_frames, yolo_model, yolo_infer_size, yolo_infer_stride, yolo_batch_size,
                      n_players_skeletons, pose_model, pose_dataset_info, pose_batch_size, update_rate_centroids=400, 
                      render_players=False, zoom_ball=False, print_time=False):
    if print_time:
        start = time.time()

    n_frames = len(original_frames)
    # Field detection
    fields, lines = field_api.detect_fields(original_frames)

    if print_time:
        print("Field detection:", time.time() - start)
        start = time.time()

    # People and ball detector
    people, ball = [], []
    for i in range(0, n_frames, yolo_batch_size):
        batch = fields[i:i+yolo_batch_size]
        batch_people, batch_ball = yolo_api.detect(yolo_model, batch, batch.shape[1:3], yolo_infer_size, yolo_infer_stride)
        #batch_people, batch_ball = our_yolo_api.detect(yolo_model, batch, half=False, model_thr=0.25)
        people += batch_people
        ball += batch_ball

    torch.cuda.empty_cache()

    if print_time:
        print("YOLO:", time.time() - start)
        start = time.time()

    # Filter players close to ball
    selected_ball = players_ball_api.choose_only_one_ball(original_frames.shape[1:3], ball)
    if n_players_skeletons > 0:
        selected_people = players_ball_api.get_n_closest_players_ball(selected_ball, people, n_players_skeletons)
    else:
        selected_people = people
    
    if print_time:
        print("Filter players close to ball:", time.time() - start)
        start = time.time()

    # Skeletons
    pose_frames = np.empty_like(original_frames)
    for i in range(0, len(selected_people), pose_batch_size):
        batch_bboxes = selected_people[i:i+pose_batch_size]
        batch_frames = fields[i:i+pose_batch_size]
        rendered_frames = pose_api.detect(pose_model, batch_frames, batch_bboxes, pose_dataset_info, render_background=render_players)
        pose_frames[i:i+pose_batch_size] = rendered_frames
    
    if print_time:
        print("Skeletons:", time.time() - start)
        start = time.time()

    # Team detection
    teams_labels = []
    kmeans = None
    for i in range(int(np.ceil(n_frames/update_rate_centroids))):
        old_centroids = kmeans.cluster_centers_ if kmeans is not None else None
        team_labels_ini, kmeans = players_ball_api.__get_teams_frame(fields[i*update_rate_centroids], people[i*update_rate_centroids], old_centroids=old_centroids)
        teams_labels += [team_labels_ini]
        teams_labels += players_ball_api.get_teams(fields[i*update_rate_centroids+1:(i+1)*update_rate_centroids], people[i*update_rate_centroids+1:(i+1)*update_rate_centroids], kmeans)
    
    if print_time:
        print("Team detection:", time.time() - start)
        start = time.time()
    
    # Painting
    painted_bboxes_over_lines = drawing_api.draw_bbox_content(original_frames=lines, list_detected_coords=people, content_frames=pose_frames)
    painted_teams_bboxes_over_lines = drawing_api.draw_teams(painted_bboxes_over_lines, people, teams_labels)
    painted_ball_teams_bboxes_over_lines = drawing_api.draw_ball(painted_teams_bboxes_over_lines, ball)  # TODO: consider using selected_ball. Depends on the postprocessing that we do with the ball

    if print_time:
        print("Painting:", time.time() - start)
        start = time.time()

    if zoom_ball:
        painted_ball_teams_bboxes_over_lines = drawing_api.zoom_at_ball(painted_ball_teams_bboxes_over_lines, selected_ball)
        if print_time:
            print("Cropping:", time.time() - start)
            start = time.time()

    return painted_ball_teams_bboxes_over_lines
