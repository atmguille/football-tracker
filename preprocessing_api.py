import time
import cv2
import numpy as np
import torch
from skimage.morphology import remove_small_holes, remove_small_objects
from sklearn.cluster import KMeans
import tensorflow.keras.backend as K

import sys
sys.path.append('./yolov7/')
sys.path.append('./mmpose/')
import yolo_api
import pose_api
import field_seg_api


TO_OPENPOSE = [0,0,5,7,9,6,8,10,11,13,15,12,14,16,1,2,3,4] #https://www.researchgate.net/figure/Openpose-18-keypoints_fig14_328857481


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


def __get_lines_canny():
    pass


def detect_fields(frames):
    fields = []
    lines = []
    for frame in frames:
        field = __get_faster_field(frame)
        fields.append(field)
        lines.append(__get_lines_tophat(field))

    return np.array(fields), np.array(lines)


def __get_statistics_detection(stat, frame, xmin, ymin, xmax, ymax):

    if stat == 'UPPER_THIRD+MEAN':
        ymax = int(ymin + (ymax-ymin)/3)

        player = frame[ymin:ymax, xmin:xmax, :]
        hsv = cv2.cvtColor(player, cv2.COLOR_RGB2HSV)
        return hsv.reshape(-1,3).mean(axis=0)

    if stat == 'ADJUSTED_SHIRT+MEAN':
        ylen = ymax-ymin
        xlen = xmax-xmin
        shirt = frame[int(ymin+0.15*ylen):int(ymin+0.5*ylen), int(xmin+0.1*xlen):int(xmin+0.9*xlen), :]
        hsv = cv2.cvtColor(shirt, cv2.COLOR_RGB2HSV)
        return hsv.reshape(-1,3).mean(axis=0)

    if stat == 'ADJUSTED_SHIRT_PANTS+MEAN':
        ylen = ymax-ymin
        xlen = xmax-xmin
        shirt = frame[int(ymin+0.15*ylen):int(ymin+0.5*ylen), int(xmin+0.1*xlen):int(xmin+0.9*xlen), :]
        pants = frame[int(ymin+0.5*ylen):int(ymin+0.75*ylen), int(xmin+0.1*xlen):int(xmin+0.9*xlen), :]
        hsv_shirt = cv2.cvtColor(shirt, cv2.COLOR_RGB2HSV)
        hsv_pants = cv2.cvtColor(pants, cv2.COLOR_RGB2HSV)
        return np.concatenate((hsv_shirt.reshape(-1,3).mean(axis=0), hsv_pants.reshape(-1,3).mean(axis=0)))


def __get_teams_frame(frame, detected_coords, kmeans=None, old_centroids=None, n_clusters=3):
    if detected_coords.shape[0] == 0 or (detected_coords.shape[0] < n_clusters and kmeans is None):
        return [], kmeans

    stat_name = 'ADJUSTED_SHIRT+MEAN'
    if stat_name == 'ADJUSTED_SHIRT_PANTS+MEAN':
        all_stats_vectors = np.zeros((len(detected_coords), 6))
    else:
        all_stats_vectors = np.zeros((len(detected_coords), 3))

    for idx, coords in enumerate(detected_coords):
        all_stats_vectors[idx] = __get_statistics_detection(stat_name, frame, *coords)

    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_stats_vectors)

        if old_centroids is not None:
            # Update the index of the new centroids based on the old ones
            # making sure that each old centroid is assigned to a different new centroid
            labels_old_centroids = kmeans.predict(old_centroids)
            if len(set(labels_old_centroids)) != len(old_centroids):
                labels_old_centroids = []
                new_centroids_copy = kmeans.cluster_centers_.tolist()
                for old_centroid in old_centroids:
                    # Get the index of the closest centroid
                    closest_centroid_idx = np.argmin(np.linalg.norm(new_centroids_copy - old_centroid, axis=1))
                    labels_old_centroids.append(closest_centroid_idx)
                    # Remove the centroid from the list of new centroids
                    new_centroids_copy[closest_centroid_idx] = [np.inf]*3

            kmeans.cluster_centers_ = kmeans.cluster_centers_[labels_old_centroids]
            teams = kmeans.predict(all_stats_vectors)
        else:
            teams = kmeans.labels_
    else:
        teams = kmeans.predict(all_stats_vectors)

    return teams, kmeans


def get_teams(frames, list_detected_coords, kmeans):
    teams_labels = []
    for frame, detected_coords in zip(frames, list_detected_coords):
        if detected_coords.size == 0:
            teams_labels.append([])
            continue

        teams, _ = __get_teams_frame(frame, detected_coords, kmeans)
        teams_labels.append(teams)

    return teams_labels


def __draw_teams_frame(frame, detected_coords, teams_labels):
    for coords, team in zip(detected_coords, teams_labels):
        xmin, ymin, xmax, ymax = coords
        if team == 0:
            color = (0, 255, 0)
        elif team == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)

    return frame


def draw_teams(frames, list_detected_coords, teams_labels):
    painted_frames = []

    for frame, detected_coords, teams in zip(frames, list_detected_coords, teams_labels):
        if detected_coords.size == 0:
            painted_frames.append(frame)
            continue

        painted_frames.append(__draw_teams_frame(frame, detected_coords, teams))
    

    return painted_frames


def choose_only_one_ball_frame(frame_shape, ball_coords):
    # In case of more than one ball, we take the one closest to the center 
    # TODO: check if this is the best option
    if len(ball_coords) > 1:
        center = np.array(frame_shape) / 2
        distances = np.linalg.norm(ball_coords[:, :2] - center, axis=1)
        return ball_coords[np.argmin(distances)]
    else:
        return ball_coords[0]


def choose_only_one_ball(frame_shape, ball_coords):
    selected_ball_coords = []

    for index, frame_ball_coords in enumerate(ball_coords):
        if frame_ball_coords.size > 0:
            selected_ball_coords.append(choose_only_one_ball_frame(frame_shape, frame_ball_coords))
        else:
            # If no ball is detected, we take the previous one
            if index > 0:
                selected_ball_coords.append(selected_ball_coords[-1])
            else:  # If it is the first frame, we take the center
                selected_ball_coords.append(np.array([frame_shape[1] / 2, frame_shape[0] / 2, frame_shape[1] / 2, frame_shape[0] / 2]))
    return selected_ball_coords


def draw_ball(frames, ball_coords, increase_factor=0.1):
    painted_frames = []
    for frame, coords in zip(frames, ball_coords):
        if coords.size == 0:
            painted_frames.append(frame)
            continue

        coords = choose_only_one_ball_frame(frame.shape[:2], coords)

        xmin, ymin, xmax, ymax = coords
        # Make ball bigger
        if increase_factor > 0:
            xmin = int(xmin - (xmax - xmin) * increase_factor)
            xmax = int(xmax + (xmax - xmin) * increase_factor)
            ymin = int(ymin - (ymax - ymin) * increase_factor)
            ymax = int(ymax + (ymax - ymin) * increase_factor)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=-1)
        painted_frames.append(frame)

    return painted_frames


def draw_bbox_content(original_frames, list_detected_coords, content_frames=None):
    # content_frames contains the content to be drawn in the bbox. If None, black

    painted_frames = []

    for idx, (original_frame, detected_coords) in enumerate(zip(original_frames, list_detected_coords)):
        if original_frame.ndim == 2:
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)

        if detected_coords.size == 0:
            painted_frames.append(original_frame)
            continue        

        for coords in detected_coords:
            xmin, ymin, xmax, ymax = coords

            player = (content_frames[idx, ymin:ymax, xmin:xmax, :] if content_frames is not None 
                      else np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8))

            
            original_frame[ymin:ymax, xmin:xmax, :] = player
        painted_frames.append(original_frame)
            
    return painted_frames

def __get_n_closest_players_ball_frame(ball_coord, detected_coords, n):
    b_xmin, b_ymin, b_xmax, b_ymax = ball_coord
    b_center = [round((b_xmax + b_xmin)/2), round((b_ymax + b_ymin)/2)]

    coords_dist = []
    
    for coords in detected_coords:
        xmin, ymin, xmax, ymax = coords
        center = [round((xmax + xmin)/2), round((ymax + ymin)/2)]
        dist = (center[0]-b_center[0])**2 + (center[1]-b_center[1])**2
        coords_dist.append([coords, dist])

    coords_dist.sort(key=lambda x: x[1])
    closest = np.array([x[0] for x in coords_dist[:n]])
    return closest

def __zoom_at_ball(frames, selected_ball, size=480):

    new_frames = []

    for k, frame in enumerate(frames):
        frame = np.pad(frame, ((size//2,size//2),(size//2,size//2),(0,0))) # We pad by size//2 so new positions = old positions +size//2 (we avoid edge balls by this)
        center = selected_ball[k]
        new_frames.append(frame[int(center[1]):int(center[1])+size, int(center[0]):int(center[0])+size, :])

    return new_frames


def __format_skeletons(skeletons, ball, max_skeles, center_at_ball=False):

    if center_at_ball:
        max_skeles -= 1

    # Converts list of (frame, n_detection, point) to (frame, 3, 18, n_detection)
    # 3 channels means height width and confidence
    output = np.zeros((len(skeletons), 3, 18, max_skeles))

    for frame, detections in enumerate(skeletons):
        for n_detection, detection in enumerate(detections):
            if n_detection >= max_skeles-(1 if not center_at_ball else 0): #-1 cause we want room for ball
                break
            for i in range(18):
                output[frame, :, i, n_detection] = detection[TO_OPENPOSE[i]]
                if center_at_ball:
                    output[frame, :2, i, n_detection] -= ball[frame][:2]

        if not center_at_ball:
            output[frame, 0, :, -1] = ball[frame][0] # TODO: change ball->graph
            output[frame, 1, :, -1] = ball[frame][1]
            output[frame, 2, :, -1] = 1 # TODO ?

    return output


def get_n_closest_players_ball(ball_coords, detected_coords, n):

    #print("Pre:", len(detected_coords), len(ball_coords))
    closest_coords = []
    for ball_coord, elem_detected_coords in zip(ball_coords, detected_coords):
        if len(ball_coord) == 0:
            # closest_coords.append(elem_detected_coords)
            if len(closest_coords) > 0:
                closest_coords.append(closest_coords[-1])
            else:
                closest_coords.append(elem_detected_coords)
        else:
            # Assumes that ball_coords is the output obtained after using choose_only_one_ball
            closest_coords.append(__get_n_closest_players_ball_frame(ball_coord, elem_detected_coords, n))
    
    #print("Post: ", len(closest_coords))
    return closest_coords


def preprocess_skeletons(original_frames, field_model, field_batch_size, yolo_model, yolo_infer_size, yolo_infer_stride, yolo_batch_size,
                      n_players_skeletons, pose_model, pose_dataset_info, pose_batch_size, update_rate_centroids, 
                      zoom_ball=False, print_time=False, device='cuda:0'):
    if print_time:
        start = time.time()

    n_frames = len(original_frames)
    # Field detection
    if field_model is None:
        fields, lines = detect_fields(original_frames)
    else:
        fields, lines = [], []
        for i in range(0, n_frames, field_batch_size):
            field_batch = original_frames[i:i+field_batch_size]
            masks = field_seg_api.detect(field_model, field_batch, resize=True, device=device)
            masks = np.expand_dims(masks*255, axis=-1)
            for frame, mask in zip(field_batch, masks):
                field = cv2.bitwise_and(frame, frame, mask=mask)
                fields.append(field)
                line = __get_lines_tophat(field)
                lines.append(line) 
        fields, lines = np.array(fields), np.array(lines)

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

    big_people = people

    if print_time:
        print("Filter small people:", time.time() - start)
        start = time.time()

    # Filter players close to ball
    selected_ball = choose_only_one_ball(original_frames.shape[1:3], ball)
    selected_people = get_n_closest_players_ball(selected_ball, big_people, n_players_skeletons)
    
    if print_time:
        print("Filter players close to ball:", time.time() - start)
        start = time.time()

    # Skeletons
    pose_frames = []
    for i in range(0, len(selected_people), pose_batch_size):
        batch_bboxes = selected_people[i:i+pose_batch_size]
        batch_frames = fields[i:i+pose_batch_size]
        rendered_frames = pose_api.detect(pose_model, batch_frames, batch_bboxes, pose_dataset_info, render_background=False, skeletons_only=True)
        pose_frames += rendered_frames

    # Converts list of (frame, n_detection, point, h, w) to (frame, 18, n_detection). +1 is for ball
    return __format_skeletons(pose_frames, selected_ball, n_players_skeletons+1, center_at_ball=zoom_ball)


def preprocess_frames(original_frames, field_model, field_batch_size, yolo_model, yolo_infer_size, yolo_infer_stride, yolo_batch_size,
                      n_players_skeletons, pose_model, pose_dataset_info, pose_batch_size, update_rate_centroids, 
                      render_players=False, zoom_ball=False, print_time=False, device='cuda:0'):
    if print_time:
        start = time.time()

    n_frames = len(original_frames)
    # Field detection
    #fields, lines = detect_fields(original_frames)
    fields, lines = [], []
    for i in range(0, n_frames, field_batch_size):
        field_batch = original_frames[i:i+field_batch_size]
        masks = field_seg_api.detect(field_model, field_batch, resize=True, device=device)
        masks = np.expand_dims(masks*255, axis=-1)
        for frame, mask in zip(field_batch, masks):
            field = cv2.bitwise_and(frame, frame, mask=mask)
            fields.append(field)
            line = __get_lines_tophat(field)
            lines.append(line)
    K.clear_session()
    fields, lines = np.array(fields), np.array(lines)

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

    """# Filter 'small' people
    mask = []
    for i, frame_preds in enumerate(people):
        mask.append([])
        for j, p in enumerate(frame_preds):
            x_diff = p[2]-p[0]
            y_diff = p[3]-p[1]
            big_enough = x_diff * y_diff > 50 and x_diff > 3 and y_diff > 3
            mask[i].append(big_enough)
    big_people = people.copy()
    for i in range(n_frames):
        big_people[i] = big_people[i][mask[i]]"""
    big_people = people

    if print_time:
        print("Filter small people:", time.time() - start)
        start = time.time()

    # Filter players close to ball
    selected_ball = choose_only_one_ball(original_frames.shape[1:3], ball)
    selected_people = get_n_closest_players_ball(selected_ball, big_people, n_players_skeletons)
    
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
        team_labels_ini, kmeans = __get_teams_frame(fields[i*update_rate_centroids], big_people[i*update_rate_centroids], old_centroids=old_centroids)
        teams_labels += [team_labels_ini]
        teams_labels += get_teams(fields[i*update_rate_centroids+1:(i+1)*update_rate_centroids], big_people[i*update_rate_centroids+1:(i+1)*update_rate_centroids], kmeans)
    
    if print_time:
        print("Team detection:", time.time() - start)
        start = time.time()
    
    # Painting
    painted_bboxes_over_lines = draw_bbox_content(original_frames=lines, list_detected_coords=big_people, content_frames=pose_frames)
    painted_teams_bboxes_over_lines = draw_teams(painted_bboxes_over_lines, big_people, teams_labels)
    painted_ball_teams_bboxes_over_lines = draw_ball(painted_teams_bboxes_over_lines, ball)  # TODO: consider using selected_ball! Depends on the postprocessing that we do with the ball

    if print_time:
        print("Painting:", time.time() - start)
        start = time.time()

    if zoom_ball:
        painted_ball_teams_bboxes_over_lines = __zoom_at_ball(painted_ball_teams_bboxes_over_lines, selected_ball)

        if print_time:
            print("Cropping:", time.time() - start)
            start = time.time()

    return painted_ball_teams_bboxes_over_lines