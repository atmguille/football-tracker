import cv2
import numpy as np
from sklearn.cluster import KMeans


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


def get_n_closest_players_ball(ball_coords, detected_coords, n):
    closest_coords = []
    for ball_coord, elem_detected_coords in zip(ball_coords, detected_coords):
        if len(ball_coord) == 0:
            if len(closest_coords) > 0:
                closest_coords.append(closest_coords[-1])
            else:
                closest_coords.append(elem_detected_coords)
        else:
            # Assumes that ball_coords is the output obtained after using choose_only_one_ball
            closest_coords.append(__get_n_closest_players_ball_frame(ball_coord, elem_detected_coords, n))
    
    return closest_coords
