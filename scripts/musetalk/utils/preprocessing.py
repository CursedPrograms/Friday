#!/usr/bin/env python3

import numpy as np
import cv2
import pickle
import os
import torch
from tqdm import tqdm

# ── face-alignment (pip install face-alignment) ───────────────────────────────
import face_alignment
from face_alignment import LandmarksType

device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable torch.compile / TorchDynamo — Triton is unavailable on Windows so
# every compile attempt fails anyway.  Setting TORCH_COMPILE_DISABLE=1 in
# dream.py covers the normal runtime path; this covers standalone execution.
import os as _os
_os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import torch._dynamo as _dynamo
_dynamo.config.suppress_errors = True
import logging as _logging
_logging.getLogger("torch._dynamo").setLevel(_logging.CRITICAL)

fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device)

# Auto-detect which landmark method this version of face_alignment exposes
if hasattr(fa, "get_landmarks_from_image"):
    _landmark_fn = fa.get_landmarks_from_image
elif hasattr(fa, "get_landmarks"):
    _landmark_fn = fa.get_landmarks
else:
    raise RuntimeError(
        "Installed face_alignment has neither get_landmarks_from_image nor "
        "get_landmarks. Run: pip install --upgrade face-alignment"
    )

print(f"[preprocessing] using face_alignment method: {_landmark_fn.__name__}  device: {device}")

# placeholder when no face is detected
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

# ── face_alignment 68-point landmark indices ──────────────────────────────────
#  0-16  : jawline
# 17-21  : left eyebrow
# 22-26  : right eyebrow
# 27-35  : nose bridge + tip
# 36-41  : left eye
# 42-47  : right eye
# 48-67  : mouth
#
# Key nose points used for vertical-split calculation:
#   [29] = just above nose tip
#   [30] = nose tip
#   [31] = just below nose tip
# ─────────────────────────────────────────────────────────────────────────────


def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm    = landmark / np.array([w, h], dtype=np.float32)
    landmark_resized = landmark_norm * np.array([new_w, new_h], dtype=np.float32)
    return landmark_resized


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        frames.append(frame)
    return frames


def _get_landmarks(frame_bgr):
    """Return 68-pt landmark list for a single BGR frame, or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _landmark_fn(frame_rgb)


def _get_bboxes(frames_bgr):
    """Return per-frame bounding boxes derived from 68-pt landmarks.

    face_alignment ≥1.4 removed get_detections_for_batch; we derive the
    coarse bbox from the landmarks that are already available via _landmark_fn.
    The bbox is only used as a fallback in get_landmark_and_bbox when the
    refined landmark-based bbox is degenerate, so precision here is fine.
    """
    bboxes = []
    for frame in frames_bgr:
        lms = _get_landmarks(frame)
        if lms is None or len(lms) == 0:
            bboxes.append(None)
        else:
            pts = lms[0].astype(np.float32)
            bboxes.append(np.array([
                pts[:, 0].min(), pts[:, 1].min(),
                pts[:, 0].max(), pts[:, 1].max(),
            ], dtype=np.float32))
    return bboxes


# ── public API ────────────────────────────────────────────────────────────────

def get_bbox_range(img_list, upperbondrange=0):
    """
    Returns a human-readable string describing the valid bbox_shift range
    for the given image list.
    """
    frames = read_imgs(img_list)
    batches = [[frames[i]] for i in range(len(frames))]

    label = f"bbox_shift: {upperbondrange}" if upperbondrange != 0 else "default value"
    print(f"get key_landmark and face bounding boxes with the {label}")

    range_minus_vals = []
    range_plus_vals  = []

    for fb in tqdm(batches):
        frame          = fb[0]
        landmarks_list = _get_landmarks(frame)
        bboxes         = _get_bboxes(fb)

        for bbox in bboxes:
            if bbox is None:
                continue
            if not landmarks_list:
                continue

            lm = landmarks_list[0].astype(np.int32)      # (68, 2)
            half_face_coord = lm[30].copy()

            range_minus_vals.append(int((lm[31] - lm[30])[1]))
            range_plus_vals.append(int((lm[30] - lm[29])[1]))

            if upperbondrange != 0:
                half_face_coord[1] += upperbondrange

    if not range_minus_vals:
        return (
            f"Total frame:[{len(frames)}] "
            f"No faces detected. current value: {upperbondrange}"
        )

    avg_minus = int(sum(range_minus_vals) / len(range_minus_vals))
    avg_plus  = int(sum(range_plus_vals)  / len(range_plus_vals))
    return (
        f"Total frame:[{len(frames)}] "
        f"Manually adjust range : [ -{avg_minus}~{avg_plus} ] , "
        f"the current value: {upperbondrange}"
    )


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Returns (coords_list, frames).
    coords_list[i] is either coord_placeholder or (x1, y1, x2, y2) ints.
    """
    frames  = read_imgs(img_list)
    batches = [[frames[i]] for i in range(len(frames))]

    label = f"bbox_shift: {upperbondrange}" if upperbondrange != 0 else "default value"
    print(f"get key_landmark and face bounding boxes with the {label}")

    coords_list      = []
    range_minus_vals = []
    range_plus_vals  = []

    for fb in tqdm(batches):
        frame          = fb[0]
        landmarks_list = _get_landmarks(frame)
        bboxes         = _get_bboxes(fb)

        for bbox in bboxes:
            if bbox is None:
                coords_list.append(coord_placeholder)
                continue

            if not landmarks_list:
                coords_list.append(tuple(int(v) for v in bbox))
                continue

            lm              = landmarks_list[0].astype(np.int32)   # (68, 2)
            half_face_coord = lm[30].copy()

            range_minus_vals.append(int((lm[31] - lm[30])[1]))
            range_plus_vals.append(int((lm[30] - lm[29])[1]))

            if upperbondrange != 0:
                half_face_coord[1] += upperbondrange

            half_face_dist = int(np.max(lm[:, 1])) - half_face_coord[1]
            upper_bond     = max(0, half_face_coord[1] - half_face_dist)

            x1 = int(np.min(lm[:, 0]))
            y1 = int(upper_bond)
            x2 = int(np.max(lm[:, 0]))
            y2 = int(np.max(lm[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                print(f"  warning: bad landmark bbox {(x1,y1,x2,y2)}, falling back to detector bbox")
                coords_list.append(tuple(int(v) for v in bbox))
            else:
                coords_list.append((x1, y1, x2, y2))

    if range_minus_vals:
        avg_minus = int(sum(range_minus_vals) / len(range_minus_vals))
        avg_plus  = int(sum(range_plus_vals)  / len(range_plus_vals))
        print("=" * 80)
        print(
            f"Total frame:[{len(frames)}] "
            f"Manually adjust range : [ -{avg_minus}~{avg_plus} ] , "
            f"the current value: {upperbondrange}"
        )
        print("=" * 80)

    return coords_list, frames


# ── smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png",
    ]
    crop_coord_path = "./coord_face.pkl"

    coords_list, full_frames = get_landmark_and_bbox(img_list)

    with open(crop_coord_path, "wb") as fh:
        pickle.dump(coords_list, fh)

    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        print("Cropped shape:", crop.shape)

    print(coords_list)