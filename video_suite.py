# -*- coding: utf-8 -*-
"""
Video Suite - automatic multi-frame burst-fusion tool

Companion to runner_suite_core.py. Where that tool picks the single sharpest
frame per runner out of a folder of pre-extracted JPEGs, this tool ingests a
raw video file directly, tracks each runner across frames with ByteTrack, and
for each runner fuses a short burst of consecutive frames around their best
moment into one higher-detail photo (dense optical flow alignment + per-pixel
confidence-weighted blending, so fast-moving limbs fall back to the single
sharp anchor frame instead of ghosting).

Required files in the same folder:
- video_suite.py
- runner_suite_core.py
- settings_video.json
- yolov8m-pose.pt (or whichever hardware.model_path resolves to)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from runner_suite_core import (
    Box,
    DEFAULT_CONFIG,
    HighResRunnerSuite,
    clamp,
    cv2,
    deep_merge,
    ensure_dir,
    np,
    tqdm,
)

VIDEO_CONFIG_EXTRA: Dict[str, Any] = {
    "video": {
        "input_video": "input_video.mp4",
        "tracker": "bytetrack.yaml",
        "burst_frames": 10,              # ~360ms @ 25fps, validated against real footage
        "max_miss_frames": 15,           # consecutive missed frames before finalizing a track
        "min_track_length_frames": 3,    # discard tracks shorter than this (noise/spurious detections)
        "max_track_length_frames": 200,  # safety cap (~8s @ 25fps) against runaway/misassociated tracks
        "fusion_scale": 2,               # upsample factor applied inside the burst fusion
        "fusion_sigma": 18.0,            # confidence falloff: exp(-residual^2/(2*sigma^2))
        "fusion_window_pad_pct": 0.25,   # padding added to the union-of-boxes crop window
        "frame_buffer_size": 90,         # ring-buffer depth in frames (~534MB worst case at 1080p; lower for 4K)
        "min_flow_window_px": 40,        # below this, skip optical flow and just Lanczos-upscale the anchor
    }
}


@dataclass
class TrackState:
    track_id: int
    first_frame_idx: int
    last_seen_frame_idx: int = -1
    miss_streak: int = 0

    # Every frame this track_id appeared in, regardless of selection-filter
    # outcome -- needed purely for burst pixel/box alignment.
    all_frames: List[int] = field(default_factory=list)
    all_boxes: Dict[int, "Box"] = field(default_factory=dict)

    # Subset that passed the selection-filter gate and got a quality score.
    candidate_frames: List[int] = field(default_factory=list)
    candidate_meta: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    best_score: float = -1.0
    anchor_frame_idx: Optional[int] = None
    # Deep copy of the anchor frame taken the moment it becomes the best
    # candidate, decoupled from ring-buffer eviction: a long-lived track can
    # still finalize correctly (possibly with a smaller burst) even if the
    # ring buffer has since evicted the neighboring frames around it.
    anchor_snapshot: Optional[Any] = None
    anchor_snapshot_box: Optional["Box"] = None


class VideoRunnerSuite(HighResRunnerSuite):
    def _load_config(self, path: Path) -> Dict[str, Any]:
        # Mirrors HighResRunnerSuite._load_config but additively merges in the
        # video-only config section, without touching runner_suite_core.py.
        # The base version returns DEFAULT_CONFIG unmerged when the file is
        # missing, so both branches need the video defaults folded in here.
        merged_defaults = deep_merge(DEFAULT_CONFIG, VIDEO_CONFIG_EXTRA)
        if not path.exists():
            print(f"[WARN] {path} not found -- using default video configuration.")
            return merged_defaults
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        return deep_merge(merged_defaults, user_cfg)

    # ------------------------------------------------------------------
    # Selection filters (duplicated from HighResRunnerSuite._process_detection_results
    # rather than touching runner_suite_core.py -- same self.cfg keys, same behavior)
    # ------------------------------------------------------------------
    def _passes_selection_filters(
        self, box: "Box", kp: Any, img_shape: Tuple[int, int]
    ) -> Tuple[bool, Optional[str]]:
        sf = self.cfg["selection_filters"]
        h_img, w_img = img_shape
        x1, y1, x2, y2 = box
        box_h, box_cx = y2 - y1, (x1 + x2) / 2.0
        is_full_frame = self._is_full_frame_runner(box, img_shape)

        if bool(sf.get("enable_safe_zone", True)) and not (
            is_full_frame and bool(self.cfg["full_frame_runner"].get("bypass_safe_zone", True))
        ):
            safe_zone = float(sf.get("safe_zone_percent", 12)) / 100.0
            if box_cx < w_img * safe_zone or box_cx > w_img * (1.0 - safe_zone):
                return False, "unsafe_edge"

        if not is_full_frame and box_h < max(
            h_img * float(sf.get("min_height_ratio", 0.15)), int(sf.get("min_box_height_px", 220))
        ):
            return False, "too_small"

        keypoint_conf = float(sf.get("keypoint_conf", 0.30))
        visible_kps = int((kp[:, 2] > keypoint_conf).sum()) if kp is not None else 0
        if not (is_full_frame and bool(self.cfg["full_frame_runner"].get("bypass_min_keypoints", True))):
            if visible_kps < int(sf.get("min_keypoints", 4)):
                return False, "too_few_keypoints"
            if bool(sf.get("require_frontal_face", False)) and kp is not None:
                if not (kp[0][2] > 0.45 and (kp[1][2] > 0.35 or kp[2][2] > 0.35)):
                    return False, "no_frontal_face"
        return True, None

    # ------------------------------------------------------------------
    # Burst fusion
    # ------------------------------------------------------------------
    def _fuse_burst(
        self,
        frames_and_boxes: List[Tuple[Any, "Box"]],
        anchor_idx: int,
        video_cfg: Dict[str, Any],
    ) -> Any:
        pad_pct = float(video_cfg.get("fusion_window_pad_pct", 0.25))
        sigma = float(video_cfg.get("fusion_sigma", 18.0))
        scale = int(video_cfg.get("fusion_scale", 2))
        min_win = int(video_cfg.get("min_flow_window_px", 40))

        boxes = [b for _, b in frames_and_boxes]
        xs1, ys1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        xs2, ys2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        w, h = xs2 - xs1, ys2 - ys1
        pad_w, pad_h = w * pad_pct, h * pad_pct

        anchor_frame, _ = frames_and_boxes[anchor_idx]
        fh, fw = anchor_frame.shape[:2]
        cx1 = int(clamp(xs1 - pad_w, 0, fw - 1))
        cy1 = int(clamp(ys1 - pad_h, 0, fh - 1))
        cx2 = int(clamp(xs2 + pad_w, cx1 + 1, fw))
        cy2 = int(clamp(ys2 + pad_h, cy1 + 1, fh))

        def window(img: Any) -> Any:
            return img[cy1:cy2, cx1:cx2]

        anchor = window(anchor_frame)
        Hh, Ww = anchor.shape[:2]

        if len(frames_and_boxes) == 1 or min(Hh, Ww) < min_win:
            return cv2.resize(anchor, (Ww * scale, Hh * scale), interpolation=cv2.INTER_LANCZOS4)

        anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
        grid_x, grid_y = np.meshgrid(np.arange(Ww), np.arange(Hh))

        aligned: List[Any] = []
        confmaps: List[Any] = []
        for i, (frame, _box) in enumerate(frames_and_boxes):
            win = window(frame)
            if win.shape[:2] != (Hh, Ww):
                win = cv2.resize(win, (Ww, Hh))
            if i == anchor_idx:
                aligned.append(anchor.astype(np.float32))
                confmaps.append(np.ones((Hh, Ww), dtype=np.float32))
                continue
            gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                anchor_gray, gray, None,
                pyr_scale=0.5, levels=4, winsize=21,
                iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
            )
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(win, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            aligned.append(warped.astype(np.float32))
            residual = np.abs(warped.astype(np.float32) - anchor.astype(np.float32)).mean(axis=2)
            confmaps.append(np.exp(-(residual ** 2) / (2 * sigma ** 2)).astype(np.float32))

        upsampled = [cv2.resize(a, (Ww * scale, Hh * scale), interpolation=cv2.INTER_LANCZOS4) for a in aligned]
        upsampled_conf = [cv2.resize(c, (Ww * scale, Hh * scale), interpolation=cv2.INTER_LINEAR) for c in confmaps]
        stack = np.stack(upsampled, axis=0)
        weights = np.stack(upsampled_conf, axis=0)[..., None]
        fused = (np.sum(stack * weights, axis=0) / np.sum(weights, axis=0)).clip(0, 255).astype(np.uint8)
        return fused

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_video(self) -> None:
        video_cfg = self.cfg["video"]
        input_path = self._resolve_path(str(video_cfg["input_video"]))
        if not input_path.exists():
            print(f"\n[ERROR] Input video not found: {input_path}")
            return
        ensure_dir(self.output_folder)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"\n[ERROR] Could not open video: {input_path}")
            return
        self._video_path = input_path
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        print(f"\n--- [VIDEO SUITE] {input_path.name} | fps={self._video_fps:.2f} | device={self.device_mode} ---")

        buffer_size = int(video_cfg.get("frame_buffer_size", 90))
        ring: Dict[int, Any] = {}
        active: Dict[int, TrackState] = {}
        frame_idx = -1

        det, sf = self.cfg["detector"], self.cfg["selection_filters"]
        track_kwargs: Dict[str, Any] = dict(
            imgsz=int(det.get("imgsz", 1024)), conf=float(sf.get("conf_threshold", 0.33)),
            iou=float(det.get("iou", 0.55)), max_det=int(det.get("max_det", 120)), verbose=False,
            tracker=str(video_cfg.get("tracker", "bytetrack.yaml")), persist=True,
        )
        if self.backend == "pytorch":
            track_kwargs["device"] = self.device_mode
            if str(self.device_mode).startswith("cuda"):
                track_kwargs["half"] = True
        else:
            track_kwargs["device"] = "cpu"

        pbar = tqdm(total=total_hint, desc="Video frames") if tqdm is not None else None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            ring[frame_idx] = frame
            h_img, w_img = frame.shape[:2]

            results = self.model.track(frame, **track_kwargs)
            r = results[0]
            seen_ids = set()

            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                boxes_np = r.boxes.xyxy.cpu().numpy()
                ids_np = r.boxes.id.cpu().numpy().astype(int)
                confs_np = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_np))
                kpts = r.keypoints.data.cpu().numpy() if (r.keypoints is not None and r.keypoints.data is not None) else None
                frame_boxes: List[Box] = [tuple(map(float, b[:4])) for b in boxes_np]

                for i, tid in enumerate(ids_np):
                    tid = int(tid)
                    seen_ids.add(tid)
                    box, conf = frame_boxes[i], float(confs_np[i])
                    kp = kpts[i] if kpts is not None else None

                    ts = active.get(tid)
                    if ts is None:
                        ts = TrackState(track_id=tid, first_frame_idx=frame_idx)
                        active[tid] = ts
                    ts.last_seen_frame_idx = frame_idx
                    ts.miss_streak = 0
                    ts.all_frames.append(frame_idx)
                    ts.all_boxes[frame_idx] = box

                    passed, _reason = self._passes_selection_filters(box, kp, (h_img, w_img))
                    if not passed:
                        continue

                    is_full_frame = self._is_full_frame_runner(box, (h_img, w_img))
                    if is_full_frame:
                        crop, crop_box = self.get_full_frame_crop(frame, box)
                    else:
                        crop_result = self.get_smart_crop(frame, kp, box)
                        if crop_result is None:
                            continue
                        crop, crop_box = crop_result

                    sharp_roi = crop if is_full_frame else self._extract_roi(frame, box)
                    sharpness = self.get_sharpness_score(sharp_roi)
                    if bool(sf.get("hard_reject_extreme_blur", True)) and sharpness < float(sf.get("min_sharpness_threshold", 80)):
                        continue

                    fence_info = self.detect_fence(crop, crop_box, box)
                    if bool(sf.get("hard_reject_fence", True)) and fence_info.get("fence_detected", False):
                        continue

                    score, quality_class, _details = self.calculate_quality_score(
                        frame, crop_box, box, kp, frame_boxes, i, sharpness, conf, fence_info, is_full_frame
                    )
                    if not self._should_save_class(quality_class, score):
                        continue
                    if quality_class == "reject":
                        quality_class = "review"

                    ts.candidate_frames.append(frame_idx)
                    ts.candidate_meta[frame_idx] = {
                        "score": score, "quality_class": quality_class, "sharpness": sharpness,
                        "conf": conf, "crop_box": crop_box, "person_box": box, "is_full_frame": is_full_frame,
                    }
                    if score > ts.best_score:
                        ts.best_score = score
                        ts.anchor_frame_idx = frame_idx
                        ts.anchor_snapshot = frame.copy()
                        ts.anchor_snapshot_box = box

            max_miss = int(video_cfg.get("max_miss_frames", 15))
            max_len = int(video_cfg.get("max_track_length_frames", 200))
            for tid, ts in list(active.items()):
                if tid not in seen_ids:
                    ts.miss_streak += 1
                track_len = frame_idx - ts.first_frame_idx + 1
                if ts.miss_streak >= max_miss or track_len >= max_len:
                    self._finalize_track(ts, ring)
                    del active[tid]

            # Bound memory: evict frames no longer needed by any active track,
            # or older than the fixed ring depth -- whichever is more aggressive.
            keep_from_active = min((t.first_frame_idx for t in active.values()), default=frame_idx + 1)
            keep_from_bound = frame_idx - buffer_size + 1
            prune_before = max(keep_from_active, keep_from_bound)
            for idx in [k for k in ring if k < prune_before]:
                del ring[idx]

            if pbar:
                pbar.update(1)

        cap.release()
        if pbar:
            pbar.close()

        for ts in active.values():
            self._finalize_track(ts, ring)

        report_path = self._write_report_csv()
        print("\n" + "=" * 72)
        print("VIDEO REPORT")
        print("=" * 72)
        for key in ["processed_crops", "quality_premium", "quality_good", "quality_review",
                    "video_tracks_discarded", "write_failed", "errors"]:
            print(f"{key:24s}: {self.stats.get(key, 0)}")
        if report_path is not None:
            print(f"CSV-Report: {report_path}")
        print("=" * 72)

    def _finalize_track(self, ts: TrackState, ring: Dict[int, Any]) -> None:
        video_cfg = self.cfg["video"]
        if ts.anchor_frame_idx is None or len(ts.all_frames) < int(video_cfg.get("min_track_length_frames", 3)):
            self._inc("video_tracks_discarded")
            return

        burst_n = int(video_cfg.get("burst_frames", 10))
        half = burst_n // 2
        anchor_pos = ts.all_frames.index(ts.anchor_frame_idx)
        lo = max(0, anchor_pos - half)
        hi = min(len(ts.all_frames), lo + burst_n)
        lo = max(0, hi - burst_n)
        window_indices = ts.all_frames[lo:hi]

        pairs: List[Tuple[Any, Box]] = []
        anchor_local_idx = None
        for idx in window_indices:
            if idx == ts.anchor_frame_idx:
                anchor_local_idx = len(pairs)
                pairs.append((ts.anchor_snapshot, ts.anchor_snapshot_box))
            elif idx in ring:
                pairs.append((ring[idx], ts.all_boxes[idx]))
            # else: evicted, not the anchor -- silently drop from burst
        if anchor_local_idx is None:
            anchor_local_idx = len(pairs)
            pairs.append((ts.anchor_snapshot, ts.anchor_snapshot_box))

        fused_crop = self._fuse_burst(pairs, anchor_local_idx, video_cfg)

        meta = ts.candidate_meta[ts.anchor_frame_idx]
        enhance_mode = str(self.cfg["image_quality"]["enhance_by_class"].get(meta["quality_class"], "upscale"))
        final = self.resize_final(fused_crop, enhance_mode=enhance_mode)
        final = self.denoise_image(final)
        final = self.sharpen_image(final)

        anchor_ms = (ts.anchor_frame_idx / self._video_fps) * 1000.0
        synthetic_path = Path(f"track{ts.track_id:05d}_{int(round(anchor_ms))}ms.jpg")
        out_path = self.write_output(
            synthetic_path, 0, final, meta["sharpness"], meta["conf"], meta["score"], meta["quality_class"], suffix="track"
        )
        if out_path is None:
            self._inc("write_failed")
            return

        self._inc("processed_crops")
        self._inc(f"quality_{meta['quality_class']}")
        self._report({
            "video": str(self._video_path), "track_id": ts.track_id,
            "anchor_frame_idx": ts.anchor_frame_idx, "anchor_timestamp_ms": round(anchor_ms, 1),
            "first_frame_idx": ts.first_frame_idx, "last_frame_idx": ts.last_seen_frame_idx,
            "track_length_frames": len(ts.all_frames), "candidate_frame_count": len(ts.candidate_frames),
            "burst_frame_count": len(pairs), "decision": "accept",
            "quality_score": meta["score"], "quality_class": meta["quality_class"],
            "sharpness": round(meta["sharpness"], 2), "confidence": round(meta["conf"], 4),
            "is_full_frame": meta["is_full_frame"], "output_path": str(out_path),
        })


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings_video.json"
    VideoRunnerSuite(config_file).run_video()
