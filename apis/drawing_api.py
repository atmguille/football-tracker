import cv2
import numpy as np

import players_ball_api


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


def zoom_at_ball(frames, selected_ball, size=480):

    new_frames = []

    for k, frame in enumerate(frames):
        frame = np.pad(frame, ((size//2,size//2),(size//2,size//2),(0,0))) # We pad by size//2 so new positions = old positions +size//2 (we avoid edge balls by this)
        center = selected_ball[k]
        new_frames.append(frame[int(center[1]):int(center[1])+size, int(center[0]):int(center[0])+size, :])

    return new_frames


def draw_ball(frames, ball_coords, increase_factor=0.1):
    painted_frames = []
    for frame, coords in zip(frames, ball_coords):
        if coords.size == 0:
            painted_frames.append(frame)
            continue

        coords = players_ball_api.choose_only_one_ball_frame(frame.shape[:2], coords)

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
