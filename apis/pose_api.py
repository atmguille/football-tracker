import torch
import numpy as np
import cv2

import sys
sys.path.append('../mmpose/')

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


@torch.no_grad()
def get_model(pose_config, pose_checkpoint, device='cuda:0'):
    model = init_pose_model(
        config=pose_config,
        checkpoint=pose_checkpoint,
        device=device,
    )

    dataset_info = DatasetInfo(model.cfg.data['test'].get('dataset_info', None))

    return model, dataset_info


def detect(model, frames: np.ndarray, bboxes: list, dataset_info, render_background=False, kpt_thr=0.3, radius=3, thickness=2, skeletons_only=False):
    ret_frames = np.empty_like(frames)
    old_indexes = {}
    n_empty = 0
    filtered_bboxes = []

    # Filter empty bboxes
    for frame_idx, frame_bboxes in enumerate(bboxes):
        if frame_bboxes.size == 0:
            ret_frames[frame_idx] = frames[frame_idx] if render_background else np.zeros_like(frames[frame_idx])
            n_empty += 1
        else:
            filtered_bboxes.append(frame_bboxes)
            old_indexes[frame_idx - n_empty] = frame_idx

    pose_results, _ = inference_top_down_pose_model(
        model,
        frames,
        filtered_bboxes,
        return_bboxes=False,
        bbox_thr=None,
        format='xyxy',
        dataset=model.cfg.data['test']['type'],
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)

    if skeletons_only:
        return pose_results

    # The vis function already creates an internal copy of the image,
    # so we create only one
    if not render_background:
        render_frame = np.zeros_like(frames[0])

    for idx, pose_result in enumerate(pose_results):
        render_frame = cv2.cvtColor(frames[old_indexes[idx]], cv2.COLOR_RGB2BGR) if render_background else render_frame

        frame = vis_pose_result(
            model,
            render_frame,
            pose_result,
            result_has_bboxes=False,
            dataset=model.cfg.data['test']['type'],
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            radius=radius,
            thickness=thickness,
            show=False,
            out_file=None)

        ret_frames[old_indexes[idx]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return ret_frames
