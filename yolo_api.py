import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox

@torch.no_grad()
def get_model(weights_path='yolov7.pt', infer_size=640, device='cuda:0', half=True):
    model = attempt_load(weights_path, map_location=device)
    infer_stride = int(model.stride.max())
    infer_size = check_img_size(infer_size, s=infer_stride)

    if half:
        model.half()

    # Run once for warmup
    model(torch.zeros(1, 3, infer_size, infer_size).to(device).type_as(next(model.parameters())))
    
    return model, infer_size, infer_stride


def _prepare_frames(frames, infer_size, infer_stride, device, half):
    # Resize and pad frames
    frames = np.array([letterbox(frame, new_shape=infer_size, stride=infer_stride)[0] for frame in frames])

    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
    frames = frames.half() if half else frames.float()
    frames /= 255.0
    return frames


@torch.no_grad()
def detect(model, frames: np.ndarray, original_shape, infer_size, infer_stride, conf_thr=0.25, iou_thr=0.45, classes=[0, 32], half=True):
    device = next(model.parameters()).device
    # Prepare frames
    frames = _prepare_frames(frames, infer_size, infer_stride, device, half)
    frames_shape = frames.shape
    # TODO: official script does warmup per image, but I think it's not necessary

    # Inference
    preds = model(frames, augment=False)[0]
    del frames
    # NMS
    preds = non_max_suppression(preds, conf_thr, iou_thr, classes=classes, agnostic=False)

    people_bboxes, ball_bboxes = [], []

    for pred in preds:
        pred[:, :4] = scale_coords(frames_shape[2:], pred[:, :4], original_shape).round()

        people_idx = pred[:, 5] == 0
        if people_idx.any():
            people_bboxes.append(pred[people_idx, :4].cpu().numpy().astype(np.int16))
        else:
            people_bboxes.append(np.array([]))
        if not people_idx.all():
            ball_bboxes.append(pred[~people_idx, :4].cpu().numpy().astype(np.int16))
        else:
            ball_bboxes.append(np.array([]))

    return people_bboxes, ball_bboxes
