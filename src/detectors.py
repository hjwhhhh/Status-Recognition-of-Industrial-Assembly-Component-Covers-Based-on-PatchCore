from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np

from .memory import MemoryConfig, NormalMemory


GREEN = (40, 210, 40)
RED = (30, 30, 230)
YELLOW = (0, 220, 255)
CYAN = (255, 220, 40)
WHITE = (245, 245, 245)
GRAY = (180, 180, 180)


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    status: str
    score: float
    label: str
    details: dict


def _safe_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _draw_label(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.72) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _circle_masks(shape: tuple[int, int], center: tuple[int, int], r: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    cx, cy = center
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    core = d2 <= int(r * 0.45) ** 2
    ring = (d2 > int(r * 0.62) ** 2) & (d2 <= int(r * 0.95) ** 2)
    return core, ring


class VideoADetector:
    """Detect the front cap surface in videoA using the image1 annotation semantics."""

    version = 9

    def __init__(self) -> None:
        self.roi = (520, 300, 1910, 760)
        self.max_faces = 4
        self.yellow_min = 0.45
        self.ok_sat_min = 92.0
        self.ng_sat_max = 112.0
        self.smooth_edge_max = 22.0
        self.smooth_lap_max = 55.0
        self.reset_temporal()

    def reset_temporal(self) -> None:
        self._tracks: list[dict] = []
        self._next_track_id = 1

    def train(self, video_path: str | Path, normal_end_sec: float = 360.0, stride_sec: float = 1.0) -> dict:
        """Calibrate only from the normal segment; image1 is used as design reference, not as runtime input."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        yellow_values: list[float] = []
        sat_values: list[float] = []
        frames = int(normal_end_sec / stride_sec)
        for i in range(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * stride_sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            for circle in self._raw_circles(frame):
                if circle[2] < 86:
                    continue
                feat = self._face_feature(frame, circle)
                if not self._is_face_candidate(feat):
                    continue
                if feat["yellow_ratio"] > 0.55 and feat["sat_mean"] > 90:
                    yellow_values.append(feat["yellow_ratio"])
                    sat_values.append(feat["sat_mean"])
        cap.release()

        if yellow_values:
            self.yellow_min = float(max(0.38, np.quantile(yellow_values, 0.03) * 0.75))
        if sat_values:
            self.ok_sat_min = float(max(82.0, np.quantile(sat_values, 0.08) * 0.80))
        return {
            "version": self.version,
            "normal_face_samples": len(yellow_values),
            "yellow_min": self.yellow_min,
            "ok_sat_min": self.ok_sat_min,
        }

    def predict(self, frame: np.ndarray) -> list[Detection]:
        scored: list[tuple[float, tuple[int, int, int], dict]] = []
        for circle in self._raw_circles(frame):
            if circle[2] < 86:
                continue
            feat = self._face_feature(frame, circle)
            if not self._is_face_candidate(feat):
                continue
            # Connected color components are stable; Hough fallback candidates are kept only if they
            # explain a real front face and survive the same one-object suppression.
            source_bonus = 45.0 if feat.get("source") == "segment" else 0.0
            quality = source_bonus + float(circle[2]) + 55.0 * max(feat["yellow_ratio"], feat["cover_ratio"]) + 22.0 * feat["ring_ratio"]
            scored.append((quality, circle, feat))

        kept: list[tuple[int, int, int, dict, float]] = []
        for group in self._group_scored_faces(scored):
            circle = self._merge_face_group(group)
            feat = self._face_feature(frame, circle)
            if not self._is_face_candidate(feat):
                _, circle, feat = max(group, key=lambda item: item[0])
            x, y, r = circle
            if any(self._same_face((x, y, r), (kx, ky, kr)) for kx, ky, kr, _, _ in kept):
                continue
            kept.append((x, y, r, feat, max(item[0] for item in group)))
            if len(kept) >= self.max_faces:
                break

        detections: list[Detection] = []
        for x, y, r, feat, _ in sorted(kept, key=lambda item: item[0]):
            status, score = self._classify_face(feat)
            detections.append(
                Detection(
                    box=(int(x - r), int(y - r), int(2 * r), int(2 * r)),
                    status=status,
                    score=score,
                    label="NG unremoved cover" if status == "NG" else "OK exposed filter",
                    details=feat | {"circle": (x, y, r)},
                )
            )
        return self._dedupe_detections(detections)

    def predict_stable(self, frame: np.ndarray) -> list[Detection]:
        """Predict with short-term track smoothing for video rendering."""
        raw = self.predict(frame)
        return self._smooth_detections(raw, frame)

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        cv2.rectangle(out, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), CYAN, 2)
        frame_status = "NG" if any(d.status == "NG" for d in detections) else ("OK" if detections else "WAIT")
        color = RED if frame_status == "NG" else (GREEN if frame_status == "OK" else YELLOW)
        _draw_label(out, f"videoA {frame_status}", (35, 62), color)
        _draw_label(out, "front face only: green=exposed, red=unremoved", (35, 100), WHITE, 0.62)
        for det in detections:
            x, y, w, h = det.box
            cx, cy, r = det.details["circle"]
            color = RED if det.status == "NG" else GREEN
            cv2.circle(out, (cx, cy), r, color, 3)
            cv2.circle(out, (cx, cy), max(8, int(r * 0.45)), YELLOW, 2)
            _draw_label(out, f"{det.status} {det.score:.2f}", (x, max(28, y - 10)), color, 0.62)
            cv2.putText(
                out,
                f"Y {det.details['yellow_ratio']:.2f} S {det.details['sat_mean']:.0f} T {det.details['texture']:.0f}",
                (x, min(frame.shape[0] - 12, y + h + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def _raw_circles(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        # Feature scoring must happen before strong suppression; otherwise a large but wrong
        # circle can suppress the actual front face of the same cartridge.
        return self._segmented_circles(frame) + self._hough_circles(frame)

    def _segmented_circles(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        x0, y0, x1, y1 = self.roi
        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        yellow = ((h >= 8) & (h <= 45) & (s >= 38) & (v >= 45)).astype(np.uint8) * 255
        cover = ((h >= 5) & (h <= 42) & (s >= 14) & (s <= 145) & (v >= 82)).astype(np.uint8) * 255
        pale_ring = ((s < 112) & (v > 92)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(yellow, cover)
        mask = cv2.bitwise_and(mask, pale_ring)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))

        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, tuple[int, int, int]]] = []
        for i in range(1, n):
            bx, by, bw, bh, area = [int(vv) for vv in stats[i]]
            if not (900 <= area <= 36000 and 36 <= bw <= 210 and 36 <= bh <= 210):
                continue
            aspect = bw / max(float(bh), 1.0)
            if not 0.55 <= aspect <= 1.82:
                continue
            cx_f, cy_f = cents[i]
            cx, cy = int(round(cx_f + x0)), int(round(cy_f + y0))
            if not (560 < cx < 1905 and 320 < cy < 735):
                continue
            fill = area / max(float(bw * bh), 1.0)
            if fill < 0.22:
                continue
            component_d = max(bw, bh)
            radius = int(np.clip(component_d * 0.82, 72, 138))
            candidates.append((float(area + 35.0 * fill + radius), (cx, cy, radius)))

        hough = self._hough_circles(frame)
        for hx, hy, hr in hough:
            if any(np.hypot(hx - cx, hy - cy) < max(92.0, 0.85 * max(hr, r)) for _, (cx, cy, r) in candidates):
                candidates.append((float(0.45 * hr), (hx, hy, hr)))

        return self._unique_circles(candidates)

    def _hough_circles(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        x0, y0, x1, y1 = self.roi
        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.8)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=92,
            param1=80,
            param2=23,
            minRadius=62,
            maxRadius=140,
        )
        if circles is None:
            return []
        raw: list[tuple[int, int, int]] = []
        for x, y, r in np.round(circles[0]).astype(int):
            x += x0
            y += y0
            if 560 < x < 1905 and 320 < y < 735 and 62 <= r <= 140:
                raw.append((int(x), int(y), int(r)))
        return raw

    def _unique_circles(self, candidates: list[tuple[float, tuple[int, int, int]]]) -> list[tuple[int, int, int]]:
        kept: list[tuple[float, tuple[int, int, int]]] = []
        for quality, circle in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(self._same_face(circle, old_circle) for _, old_circle in kept):
                continue
            kept.append((quality, circle))
        return [circle for _, circle in sorted(kept, key=lambda item: item[1][0])]

    def _group_scored_faces(
        self, scored: list[tuple[float, tuple[int, int, int], dict]]
    ) -> list[list[tuple[float, tuple[int, int, int], dict]]]:
        groups: list[list[tuple[float, tuple[int, int, int], dict]]] = []
        for item in sorted(scored, key=lambda value: value[0], reverse=True):
            _, circle, _ = item
            matched = None
            for idx, group in enumerate(groups):
                if any(self._same_face(circle, old_circle) for _, old_circle, _ in group):
                    matched = idx
                    break
            if matched is None:
                groups.append([item])
            else:
                groups[matched].append(item)
        groups.sort(key=lambda group: max(item[0] for item in group), reverse=True)
        return groups

    def _merge_face_group(self, group: list[tuple[float, tuple[int, int, int], dict]]) -> tuple[int, int, int]:
        ordered = sorted(group, key=lambda item: item[0], reverse=True)
        best = max(ordered[0][0], 1.0)
        usable = [item for item in ordered if item[0] >= best * 0.58][:5]
        if not usable:
            usable = ordered[:1]
        weights = np.array([max(item[0], 1.0) for item in usable], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-6)
        xs = np.array([item[1][0] for item in usable], dtype=np.float32)
        ys = np.array([item[1][1] for item in usable], dtype=np.float32)
        rs = np.array([item[1][2] for item in usable], dtype=np.float32)
        cx = int(round(float((xs * weights).sum())))
        cy = int(round(float((ys * weights).sum())))
        weighted_r = float((rs * weights).sum())
        outer_r = float(np.quantile(rs, 0.72))
        radius = int(np.clip(0.45 * weighted_r + 0.55 * outer_r, 86, 140))
        return cx, cy, radius

    def _same_face(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
        ax, ay, ar = a
        bx, by, br = b
        dist = float(np.hypot(ax - bx, ay - by))
        if dist < max(112.0, 1.12 * max(ar, br)):
            return True
        ax1, ay1, ax2, ay2 = ax - ar, ay - ar, ax + ar, ay + ar
        bx1, by1, bx2, by2 = bx - br, by - br, bx + br, by + br
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float((ax2 - ax1) * (ay2 - ay1))
        area_b = float((bx2 - bx1) * (by2 - by1))
        return inter / max(min(area_a, area_b), 1.0) > 0.30

    def _face_feature(self, frame: np.ndarray, circle: tuple[int, int, int]) -> dict:
        x, y, r = circle
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = _safe_rect(x - r, y - r, x + r, y + r, width, height)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return {}
        core, ring = _circle_masks(patch.shape[:2], (x - x1, y - y1), r)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        core_count = max(int(core.sum()), 1)
        ring_count = max(int(ring.sum()), 1)
        yellow = ((h >= 8) & (h <= 45) & (s >= 38) & (v >= 45) & core).sum() / core_count
        cover = ((h >= 5) & (h <= 42) & (s >= 14) & (s <= 135) & (v >= 85) & core).sum() / core_count
        pale = ((s < 70) & (v > 105) & core).sum() / core_count
        ring_ratio = ((s < 105) & (v > 90) & ring).sum() / ring_count
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture = float(s[core].std() + v[core].std() + edge[core].mean() + lap[core].var() / 10.0)
        return {
            "yellow_ratio": float(yellow),
            "cover_ratio": float(cover),
            "pale_ratio": float(pale),
            "ring_ratio": float(ring_ratio),
            "sat_mean": float(s[core].mean()),
            "sat_std": float(s[core].std()),
            "value_mean": float(v[core].mean()),
            "value_std": float(v[core].std()),
            "edge_mean": float(edge[core].mean()),
            "lap_var": float(lap[core].var()),
            "texture": texture,
        }

    def _is_face_candidate(self, feat: dict) -> bool:
        if not feat:
            return False
        front_ring = feat["ring_ratio"] > 0.68
        colored_center = feat["yellow_ratio"] > self.yellow_min and feat["sat_mean"] > 34.0 and front_ring
        inner_exposed_disk = feat["yellow_ratio"] > 0.70 and feat["sat_mean"] > self.ok_sat_min and front_ring
        smooth_cover = feat["cover_ratio"] > 0.76 and feat["sat_mean"] > 28.0 and feat["value_mean"] > 95.0
        return inner_exposed_disk or (feat["ring_ratio"] > 0.42 and (colored_center or smooth_cover))

    def _classify_face(self, feat: dict) -> tuple[str, float]:
        smooth = (
            feat["edge_mean"] < self.smooth_edge_max
            and feat["lap_var"] < self.smooth_lap_max
            and feat["value_std"] < 18.0
        )
        high_yellow_exposed = feat["yellow_ratio"] > 0.78 and feat["cover_ratio"] < 0.66 and feat["pale_ratio"] < 0.46
        pale_shell_ok = feat["pale_ratio"] > 0.58 and feat["cover_ratio"] > 0.70 and feat["sat_mean"] < 88.0
        unremoved_like = (
            smooth
            and feat["cover_ratio"] > 0.72
            and 28.0 < feat["sat_mean"] < self.ng_sat_max
            and feat["yellow_ratio"] > 0.55
            and not high_yellow_exposed
            and not pale_shell_ok
        )
        textured_exposed = feat["texture"] > 62.0 or feat["edge_mean"] > self.smooth_edge_max * 0.92 or feat["lap_var"] > self.smooth_lap_max * 1.08
        ok_like = (
            pale_shell_ok
            or (
                feat["yellow_ratio"] > self.yellow_min
                and not unremoved_like
                and (feat["sat_mean"] >= self.ok_sat_min or textured_exposed or high_yellow_exposed)
            )
        )
        if unremoved_like or not ok_like:
            score = (
                1.8 * feat["cover_ratio"]
                + 0.015 * max(0.0, self.ng_sat_max - feat["sat_mean"])
                + 0.018 * max(0.0, self.smooth_edge_max - feat["edge_mean"])
                + 0.010 * max(0.0, self.smooth_lap_max - feat["lap_var"])
            )
            return "NG", float(score)
        score = feat["yellow_ratio"] + feat["sat_mean"] / 255.0 + min(feat["texture"], 220.0) / 220.0
        return "OK", float(score)

    def _smooth_detections(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Detection]:
        unmatched_tracks = set(range(len(self._tracks)))
        assignments: list[tuple[int, int]] = []
        for det_idx, det in enumerate(detections):
            cx, cy, r = det.details["circle"]
            best_idx = None
            best_dist = float("inf")
            for tr_idx in unmatched_tracks:
                tr = self._tracks[tr_idx]
                pred_x = tr["cx"] + float(np.clip(tr.get("vx", 0.0), -24.0, 24.0))
                pred_y = tr["cy"] + float(np.clip(tr.get("vy", 0.0), -14.0, 14.0))
                dist = float(np.hypot(cx - pred_x, cy - pred_y))
                gate = max(145.0, 1.25 * max(r, tr["r"]))
                if dist < gate and dist < best_dist:
                    best_dist = dist
                    best_idx = tr_idx
            if best_idx is not None:
                unmatched_tracks.remove(best_idx)
                assignments.append((best_idx, det_idx))

        matched_dets = {det_idx for _, det_idx in assignments}
        output_tracks: list[dict] = []
        for tr_idx, det_idx in assignments:
            tr = self._tracks[tr_idx]
            det = detections[det_idx]
            raw_x, raw_y, raw_r = det.details["circle"]
            old_x, old_y = tr["cx"], tr["cy"]
            base_x = old_x + float(np.clip(tr.get("vx", 0.0), -22.0, 22.0))
            base_y = old_y + float(np.clip(tr.get("vy", 0.0), -8.0, 8.0))
            raw_jump = float(np.hypot(raw_x - base_x, raw_y - base_y))
            if raw_jump <= 30.0:
                alpha = 0.10
            elif raw_jump <= 120.0:
                alpha = 0.24
            else:
                alpha = 0.08 if tr["age"] >= 4 else 0.20
            tr["cx"] = (1.0 - alpha) * base_x + alpha * raw_x
            tr["cy"] = (1.0 - alpha) * base_y + alpha * raw_y
            tr["r"] = 0.99 * tr["r"] + 0.01 * raw_r
            mvx = tr["cx"] - old_x
            mvy = tr["cy"] - old_y
            old_vx = tr.get("vx", 0.0)
            old_vy = tr.get("vy", 0.0)
            tr["vx"] = 0.80 * old_vx + 0.20 * mvx
            tr["vy"] = 0.86 * old_vy + 0.14 * mvy
            if mvx * old_vx < 0 or abs(mvx) < 1.0:
                tr["vx"] *= 0.25
            if mvy * old_vy < 0 or abs(mvy) < 1.0:
                tr["vy"] *= 0.25
            tr["status"] = det.status
            tr["score"] = 0.75 * tr.get("score", det.score) + 0.25 * det.score
            tr["label"] = det.label
            tr["details"] = det.details
            tr["missed"] = 0
            tr["age"] += 1
            output_tracks.append(tr)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            cx, cy, r = det.details["circle"]
            if any(self._same_face((cx, cy, r), (int(round(tr["cx"])), int(round(tr["cy"])), int(round(tr["r"])))) for tr in output_tracks):
                continue
            tr = {
                "id": self._next_track_id,
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r),
                "vx": 0.0,
                "vy": 0.0,
                "status": det.status,
                "score": det.score,
                "label": det.label,
                "details": det.details,
                "missed": 0,
                "age": 1,
            }
            self._next_track_id += 1
            self._tracks.append(tr)
            output_tracks.append(tr)

        for tr_idx in list(unmatched_tracks):
            tr = self._tracks[tr_idx]
            tr["vx"] = 0.35 * tr.get("vx", 0.0)
            tr["vy"] = 0.35 * tr.get("vy", 0.0)
            tr["missed"] += 1
            if tr["missed"] <= 8 and tr["age"] >= 3 and tr["r"] >= 86:
                near_exit = tr["cx"] < self.roi[0] + 10 or tr["cx"] > self.roi[2] - 10
                if near_exit and tr["missed"] > 2:
                    continue
                if frame is not None and tr["missed"] > 4:
                    circle = (int(round(tr["cx"])), int(round(tr["cy"])), int(round(tr["r"])))
                    feat = self._face_feature(frame, circle)
                    if not self._is_face_candidate(feat):
                        continue
                    status, score = self._classify_face(feat)
                    tr["status"] = status
                    tr["score"] = 0.65 * tr.get("score", score) + 0.35 * score
                    tr["label"] = "NG unremoved cover" if status == "NG" else "OK exposed filter"
                    tr["details"] = feat | {"circle": circle}
                output_tracks.append(tr)

        self._tracks = [tr for tr in self._tracks if tr["missed"] <= 12 and 430 < tr["cx"] < 1980 and 220 < tr["cy"] < 850]
        stable: list[Detection] = []
        for tr in sorted(output_tracks, key=lambda item: item["cx"]):
            if tr["age"] < 2:
                continue
            cx, cy, r = int(round(tr["cx"])), int(round(tr["cy"])), int(round(tr["r"]))
            details = dict(tr["details"])
            details["circle"] = (cx, cy, r)
            stable.append(
                Detection(
                    box=(cx - r, cy - r, 2 * r, 2 * r),
                    status=tr["status"],
                    score=float(tr["score"]),
                    label=tr["label"],
                    details=details,
                )
            )
        return self._dedupe_detections(stable)

    def _dedupe_detections(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for det in sorted(detections, key=lambda item: item.score, reverse=True):
            circle = det.details.get("circle")
            if circle is None:
                continue
            if circle[2] < 86:
                continue
            if any(self._same_face(circle, old.details["circle"]) for old in kept):
                continue
            kept.append(det)
        if len(kept) > self.max_faces:
            kept = sorted(kept, key=lambda item: (item.details["circle"][2], item.score), reverse=True)[: self.max_faces]
        return sorted(kept, key=lambda item: item.details["circle"][0])


class VideoBDetector:
    """Check the reference cap sites in videoB with geometry-locked side-cap sites."""

    version = 21

    def __init__(self) -> None:
        self.model_signature = "videoB_removed_side_ng_v21"
        self.right_x0 = 900
        self.min_center_x = 1025
        self.side_upper_offset = -95
        self.side_lower_offset = -76
        self.side_upper_y = 415
        self.side_lower_y = 870
        self.side_lower_far_x_min = 1760.0
        self.side_lower_far_offset = -116
        self.side_lower_far_y = 888
        self.side_lower_far_half = (44, 44)
        self.nominal_pitch = 305.0
        self.require_direct_side = True
        self.side_missing_grace = 10
        self.side_gray_ng_min = 1850.0
        self.side_edge_ng_min = 18.0
        self.side_present_min = 0.52
        self.side_removed_edge_min = 88.0
        self.side_removed_value_std_min = 43.0
        self.memories = {
            "top": NormalMemory(MemoryConfig(max_memory=2000, threshold_quantile=0.99, threshold_scale=1.35)),
            "bottom": NormalMemory(MemoryConfig(max_memory=2000, threshold_quantile=0.99, threshold_scale=1.35)),
            "side": NormalMemory(MemoryConfig(max_memory=2500, threshold_quantile=0.99, threshold_scale=1.45)),
        }
        self.blue_min = {"top": 1200.0, "bottom": 1200.0}
        self.gray_min = 900.0
        self.skip_occluded = True
        self.reset_temporal()

    def reset_temporal(self) -> None:
        self._center_tracks: list[dict] = []
        self._next_center_id = 1
        self._site_tracks: list[dict] = []
        self._next_site_id = 1

    def train(self, video_path: str | Path, normal_end_sec: float = 180.0, stride_sec: float = 1.0) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        samples: dict[str, list[np.ndarray]] = {"top": [], "bottom": [], "side": []}
        blue_values: dict[str, list[float]] = {"top": [], "bottom": []}
        gray_values: list[float] = []
        side_presence_values: list[float] = []
        frames = int(normal_end_sec / stride_sec)
        for i in range(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * stride_sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            for site in self._expected_sites(frame):
                if site.get("side_missing_expected"):
                    continue
                feat, meta = self._site_feature(frame, site)
                if meta["skip"]:
                    continue
                group = "side" if site["kind"].startswith("side") else site["kind"]
                samples[group].append(feat)
                if group in blue_values:
                    blue_values[group].append(meta["blue_area"])
                else:
                    gray_values.append(meta["gray_area"])
                    side_presence_values.append(meta.get("side_presence", 0.0))
        cap.release()

        info = {"version": self.version, "model_signature": self.model_signature}
        for group, values in samples.items():
            if len(values) < 8:
                continue
            self.memories[group].fit(np.vstack(values))
            info[f"{group}_samples"] = len(values)
            info[f"{group}_threshold"] = float(self.memories[group].threshold)
        for group in ["top", "bottom"]:
            if blue_values[group]:
                self.blue_min[group] = float(max(650.0, np.quantile(blue_values[group], 0.05) * 0.45))
                info[f"{group}_blue_min"] = self.blue_min[group]
        if gray_values:
            self.gray_min = float(max(260.0, np.quantile(gray_values, 0.05) * 0.35))
            info["side_gray_min"] = self.gray_min
        if side_presence_values:
            self.side_present_min = float(max(0.48, min(0.66, np.quantile(side_presence_values, 0.05) * 0.72)))
            info["side_present_min"] = self.side_present_min
        return info

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return self._predict_from_sites(frame, self._expected_sites(frame, use_history=False))

    def predict_stable(self, frame: np.ndarray) -> list[Detection]:
        sites = self._expected_sites(frame, use_history=True)
        return self._predict_from_sites(frame, sites)

    def _predict_from_sites(self, frame: np.ndarray, sites: list[dict]) -> list[Detection]:
        detections: list[Detection] = []
        evaluated: list[tuple[dict, np.ndarray, dict, str]] = []
        main_visible: dict[int, int] = {}
        for site in sites:
            feat, meta = self._site_feature(frame, site)
            group = "side" if site["kind"].startswith("side") else site["kind"]
            evaluated.append((site, feat, meta, group))
            if group in ("top", "bottom") and not meta["skip"]:
                if meta["blue_area"] >= self.blue_min[group] * 0.70:
                    tube_idx = int(site.get("tube_index", -1))
                    main_visible[tube_idx] = main_visible.get(tube_idx, 0) + 1

        for site, feat, meta, group in evaluated:
            if meta["skip"]:
                detections.append(
                    Detection(
                        box=site["box_xywh"],
                        status="SKIP",
                        score=0.0,
                        label="SKIP occluded/out of frame",
                        details=site | meta,
                    )
                )
                continue

            generated_center = "history" in site.get("center_source", "") or "interpolated" in site.get("center_source", "")
            near_left_edge = site["center"][0] < self.min_center_x + 55
            tube_idx = int(site.get("tube_index", -1))
            if group == "side":
                if (generated_center or near_left_edge) and main_visible.get(tube_idx, 0) == 0:
                    detections.append(
                        Detection(
                            box=site["box_xywh"],
                            status="SKIP",
                            score=0.0,
                            label="SKIP side parent not visible",
                            details=site | meta | {"group": group, "reason": "side_parent_not_visible"},
                        )
                    )
                    continue
                presence = float(meta.get("side_presence", 0.0))
                removed_texture = (
                    site["kind"] == "side_lower"
                    and float(meta.get("edge_mean", 0.0)) >= self.side_removed_edge_min
                    and float(meta.get("value_std", 0.0)) >= self.side_removed_value_std_min
                )
                if removed_texture:
                    status = "NG"
                    side_state = "removed"
                    score = float(meta.get("edge_mean", 0.0)) / 100.0
                    reason = "side_cap_removed_by_texture"
                elif presence >= self.side_present_min:
                    status = "OK"
                    side_state = "capped"
                    score = presence
                    reason = "side_cap_capped"
                else:
                    status = "NG"
                    side_state = "open"
                    score = max(0.0, self.side_present_min - presence)
                    reason = "side_cap_open"
                detections.append(
                    Detection(
                        box=site["box_xywh"],
                        status=status,
                        score=float(score),
                        label=f"{site['kind']} {side_state}",
                        details=site
                        | meta
                        | {
                            "group": group,
                            "side_state": side_state,
                            "removed_texture": bool(removed_texture),
                            "reason": reason,
                        },
                    )
                )
                continue

            if group in ("top", "bottom") and (generated_center or near_left_edge) and meta["blue_area"] < self.blue_min[group] * 1.25:
                detections.append(
                    Detection(
                        box=site["box_xywh"],
                        status="SKIP",
                        score=0.0,
                        label="SKIP generated main cap not visible",
                        details=site | meta | {"group": group, "reason": "generated_main_cap_not_visible"},
                    )
                )
                continue
            memory = self.memories.get(group)
            if memory is not None and memory.threshold is not None:
                score = float(memory.score(feat.reshape(1, -1))[0])
                threshold_ng = score > float(memory.threshold)
            else:
                score = 0.0
                threshold_ng = False
            if group in ("top", "bottom"):
                rule_ng = meta["blue_area"] < self.blue_min[group]
                memory_ng = threshold_ng and meta["blue_area"] < self.blue_min[group] * 1.35
            else:
                rule_ng = False
                memory_ng = False
            status = "NG" if rule_ng or memory_ng else "OK"
            detections.append(
                Detection(
                    box=site["box_xywh"],
                    status=status,
                    score=score,
                    label=f"{site['kind']} {status}",
                    details=site | meta | {"group": group},
                )
            )
        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        cv2.line(out, (self.right_x0, 0), (self.right_x0, frame.shape[0]), CYAN, 2)
        valid = [d for d in detections if d.status != "SKIP"]
        frame_status = "NG" if any(d.status == "NG" for d in valid) else ("OK" if valid else "WAIT")
        color = RED if frame_status == "NG" else (GREEN if frame_status == "OK" else YELLOW)
        _draw_label(out, f"videoB {frame_status}", (35, 62), color)
        _draw_label(out, "side cap boxes are locked to the red-reference positions", (35, 100), WHITE, 0.62)
        for det in detections:
            x, y, w, h = det.box
            if det.status == "SKIP":
                draw_color = GRAY
            else:
                draw_color = RED if det.status == "NG" else GREEN
            thickness = 3 if det.details.get("direct_side") or det.status == "NG" else 2
            cv2.rectangle(out, (x, y), (x + w, y + h), draw_color, thickness)
            if det.status == "SKIP":
                label = "SKIP"
            elif det.details.get("group") == "side":
                label = "CAPPED" if det.status == "OK" else ("REMOVED" if det.details.get("removed_texture") else "OPEN")
            else:
                label = f"{det.status} {det.score:.1f}"
            cv2.putText(out, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, draw_color, 2, cv2.LINE_AA)
        return out

    def _detect_tubes(self, frame: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
        roi = frame[:, self.right_x0 :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        label_mask = (((h >= 92) & (h <= 172) & (s >= 52) & (v >= 35))).astype(np.uint8) * 255
        label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21)))
        label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
        n, _, stats, cents = cv2.connectedComponentsWithStats(label_mask, 8)
        found: list[tuple[float, tuple[int, int, int, int]]] = []
        for i in range(1, n):
            x, y, w, hgt, area = [int(vv) for vv in stats[i]]
            cx, _ = cents[i]
            # A hand-held or tilted cartridge produces a wide/short label component.
            # The right-lane inspection objects are upright and their label is tall.
            if area > 2600 and 320 <= hgt <= 540 and 18 <= w <= 92 and 340 <= y <= 600 and cx + self.right_x0 >= self.min_center_x:
                found.append((float(cx + self.right_x0), (x + self.right_x0, y, w, hgt)))
        found.sort(key=lambda item: item[0])
        merged: list[tuple[float, tuple[int, int, int, int]]] = []
        for cx, box in found:
            if not merged or abs(cx - merged[-1][0]) > 82:
                merged.append((cx, box))
            else:
                old_cx, old_box = merged[-1]
                x1 = min(old_box[0], box[0])
                y1 = min(old_box[1], box[1])
                x2 = max(old_box[0] + old_box[2], box[0] + box[2])
                y2 = max(old_box[1] + old_box[3], box[1] + box[3])
                merged[-1] = ((old_cx + cx) / 2.0, (x1, y1, x2 - x1, y2 - y1))
        return merged

    def _detect_main_cap_pairs(self, frame: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (90, 45, 30), (135, 255, 255))
        blue[:, : self.right_x0] = 0
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(blue, 8)
        top: list[tuple[float, tuple[int, int, int, int], int]] = []
        bottom: list[tuple[float, tuple[int, int, int, int], int]] = []
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            cx, cy = cents[i]
            if area < 420 or not (24 <= w <= 175 and 8 <= h <= 88):
                continue
            if 245 <= cy <= 415:
                top.append((float(cx), (x, y, w, h), area))
            elif 845 <= cy <= 1040:
                bottom.append((float(cx), (x, y, w, h), area))

        pairs: list[tuple[float, tuple[int, int, int, int]]] = []
        used_bottom: set[int] = set()
        for tx, tbox, tarea in sorted(top, key=lambda item: item[2], reverse=True):
            best_idx = None
            best_dist = 999.0
            for idx, (bx, bbox, barea) in enumerate(bottom):
                if idx in used_bottom:
                    continue
                dist = abs(tx - bx)
                if dist < 78 and dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if best_idx is None:
                continue
            used_bottom.add(best_idx)
            bx, bbox, barea = bottom[best_idx]
            cx = (tx * tarea + bx * barea) / max(tarea + barea, 1)
            x1 = int(min(tbox[0], bbox[0], cx - 48))
            y1 = int(min(tbox[1], bbox[1], 460))
            x2 = int(max(tbox[0] + tbox[2], bbox[0] + bbox[2], cx + 48))
            y2 = int(max(tbox[1] + tbox[3], bbox[1] + bbox[3], 840))
            if cx >= self.min_center_x:
                pairs.append((float(cx), (x1, y1, x2 - x1, y2 - y1)))
        return sorted(pairs, key=lambda item: item[0])

    def _detect_side_caps(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        bands = [
            ("side_upper", 345, 465),
            ("side_lower", 795, 930),
        ]
        found: list[dict] = []
        for kind, y1, y2 in bands:
            x1, x2 = max(self.right_x0, self.min_center_x - 80), min(w, 1860)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            gray_mask = (((ss < 92) & (vv > 42) & (vv < 245))).astype(np.uint8) * 255
            gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

            edges = cv2.Canny(gray, 45, 130)
            circles = cv2.HoughCircles(
                cv2.GaussianBlur(gray, (5, 5), 1.1),
                cv2.HOUGH_GRADIENT,
                dp=1.15,
                minDist=42,
                param1=95,
                param2=14,
                minRadius=14,
                maxRadius=42,
            )
            candidate_centers: list[tuple[float, float, float, float]] = []
            if circles is not None:
                for cx, cy, r in np.round(circles[0]).astype(int):
                    wx, wy = float(cx + x1), float(cy + y1)
                    if not (self.min_center_x <= wx <= w - 24):
                        continue
                    rr = int(np.clip(r * 1.35, 28, 48))
                    px1, py1, px2, py2 = _safe_rect(int(cx - rr), int(cy - rr), int(cx + rr), int(cy + rr), crop.shape[1], crop.shape[0])
                    if px2 <= px1 or py2 <= py1:
                        continue
                    g_area = float(cv2.countNonZero(gray_mask[py1:py2, px1:px2]))
                    e_area = float(cv2.countNonZero(edges[py1:py2, px1:px2]))
                    area = float(max((px2 - px1) * (py2 - py1), 1))
                    quality = g_area / area + 0.45 * e_area / area + min(float(r), 42.0) / 84.0
                    candidate_centers.append((wx, wy, float(np.clip(rr, 32, 52)), quality))

            n, _, stats, cents = cv2.connectedComponentsWithStats(gray_mask, 8)
            for i in range(1, n):
                bx, by, bw, bh, area = [int(vv) for vv in stats[i]]
                if not (280 <= area <= 6200 and 20 <= bw <= 96 and 20 <= bh <= 96):
                    continue
                aspect = bw / max(float(bh), 1.0)
                if not 0.48 <= aspect <= 2.25:
                    continue
                cx, cy = cents[i]
                wx, wy = float(cx + x1), float(cy + y1)
                if not (self.min_center_x <= wx <= w - 24):
                    continue
                rr = float(np.clip(max(bw, bh) * 0.62, 30, 52))
                quality = float(area) / max(float(bw * bh), 1.0) + min(float(area), 5000.0) / 5000.0
                candidate_centers.append((wx, wy, rr, quality))

            for wx, wy, rr, quality in sorted(candidate_centers, key=lambda item: item[3], reverse=True):
                if quality < 0.28:
                    continue
                if any(item["kind"] == kind and np.hypot(wx - item["center"][0], wy - item["center"][1]) < 62 for item in found):
                    continue
                hw = int(np.clip(rr * 1.15, 38, 58))
                hh_box = int(np.clip(rr * 1.10, 38, 58))
                sx1, sy1, sx2, sy2 = _safe_rect(int(wx - hw), int(wy - hh_box), int(wx + hw), int(wy + hh_box), w, h)
                found.append(
                    {
                        "kind": kind,
                        "center": (float(wx), float(wy)),
                        "box": (sx1, sy1, sx2, sy2),
                        "box_xywh": (sx1, sy1, sx2 - sx1, sy2 - sy1),
                        "complete": sx1 >= self.right_x0 and (sx2 - sx1) >= int(hw * 1.45) and (sy2 - sy1) >= int(hh_box * 1.45),
                        "quality": float(quality),
                    }
                )
        return sorted(found, key=lambda item: (item["kind"], item["center"][0]))

    def _infer_container_centers(self, frame: np.ndarray) -> list[dict]:
        candidates: list[dict] = []
        for cx, box in self._detect_tubes(frame):
            candidates.append({"cx": float(cx), "tube_box": box, "source": "label", "weight": 3.0})
        for cx, box in self._detect_main_cap_pairs(frame):
            candidates.append({"cx": float(cx), "tube_box": box, "source": "main_caps", "weight": 2.0})
        return self._merge_center_candidates(candidates)

    def _merge_center_candidates(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        groups: list[list[dict]] = []
        for cand in sorted(candidates, key=lambda item: item["cx"]):
            if not groups or abs(cand["cx"] - np.mean([item["cx"] for item in groups[-1]])) > 92:
                groups.append([cand])
            else:
                groups[-1].append(cand)
        centers: list[dict] = []
        for group in groups:
            weight_sum = sum(item["weight"] for item in group)
            cx = sum(item["cx"] * item["weight"] for item in group) / max(weight_sum, 1e-6)
            primary = max(group, key=lambda item: item["weight"])
            centers.append(
                {
                    "cx": float(cx),
                    "tube_box": primary["tube_box"],
                    "source": "+".join(sorted({item["source"] for item in group})),
                    "weight": weight_sum,
                }
            )
        return sorted(centers, key=lambda item: item["cx"])

    def _fill_center_gaps(self, centers: list[dict], width: int) -> list[dict]:
        if len(centers) < 2:
            return centers
        diffs = np.diff([item["cx"] for item in centers])
        valid = [float(d) for d in diffs if 225 <= d <= 390]
        pitch = float(np.median(valid)) if valid else self.nominal_pitch
        filled: list[dict] = []
        for left, right in zip(centers, centers[1:]):
            filled.append(left)
            gap = right["cx"] - left["cx"]
            missing = int(round(gap / pitch)) - 1
            if missing <= 0:
                continue
            step = gap / (missing + 1)
            if not 225 <= step <= 390:
                continue
            for idx in range(missing):
                cx = left["cx"] + step * (idx + 1)
                if self.min_center_x < cx < width - 35:
                    filled.append(
                        {
                            "cx": float(cx),
                            "tube_box": (int(cx - 36), 470, 72, 370),
                            "source": "interpolated",
                            "weight": 0.75,
                        }
                    )
        filled.append(centers[-1])
        return sorted(filled, key=lambda item: item["cx"])

    def _stabilize_centers(self, centers: list[dict], width: int) -> list[dict]:
        unmatched_tracks = set(range(len(self._center_tracks)))
        assignments: list[tuple[int, int]] = []
        for idx, center in enumerate(centers):
            best = None
            best_dist = float("inf")
            for tr_idx in unmatched_tracks:
                tr = self._center_tracks[tr_idx]
                pred = tr["cx"] + float(np.clip(tr.get("vx", 0.0), -28.0, 28.0))
                dist = abs(center["cx"] - pred)
                if dist < 115 and dist < best_dist:
                    best = tr_idx
                    best_dist = dist
            if best is not None:
                unmatched_tracks.remove(best)
                assignments.append((best, idx))

        matched_centers = {idx for _, idx in assignments}
        output: list[dict] = []
        for tr_idx, center_idx in assignments:
            tr = self._center_tracks[tr_idx]
            center = centers[center_idx]
            old = tr["cx"]
            delta = center["cx"] - old
            if abs(delta) <= 7.0:
                alpha = 0.16
            elif abs(delta) <= 72.0:
                alpha = 0.40 if center["source"] != "interpolated" else 0.25
            else:
                alpha = 0.18 if tr["age"] >= 4 else 0.34
            tr["cx"] = (1.0 - alpha) * old + alpha * center["cx"]
            measured_vx = tr["cx"] - old
            old_vx = tr.get("vx", 0.0)
            tr["vx"] = 0.64 * old_vx + 0.36 * measured_vx
            if measured_vx * old_vx < 0 or abs(measured_vx) < 0.8:
                tr["vx"] = 0.0
            tr["tube_box"] = center["tube_box"]
            tr["source"] = center["source"]
            tr["missed"] = 0
            tr["age"] += 1
            output.append(dict(center, cx=float(tr["cx"]), source=center["source"] + "+track"))

        for idx, center in enumerate(centers):
            if idx in matched_centers:
                continue
            tr = {
                "id": self._next_center_id,
                "cx": center["cx"],
                "vx": 0.0,
                "tube_box": center["tube_box"],
                "source": center["source"],
                "missed": 0,
                "age": 1,
            }
            self._next_center_id += 1
            self._center_tracks.append(tr)
            output.append(center)

        for tr_idx in list(unmatched_tracks):
            tr = self._center_tracks[tr_idx]
            tr["vx"] = 0.25 * tr.get("vx", 0.0)
            tr["missed"] += 1
            if tr["missed"] <= 2 and self.min_center_x < tr["cx"] < width - 35:
                output.append(
                    {
                        "cx": float(tr["cx"]),
                        "tube_box": tr["tube_box"],
                        "source": "history",
                        "weight": 0.5,
                    }
                )

        self._center_tracks = [
            tr
            for tr in self._center_tracks
            if tr["missed"] <= 4 and self.min_center_x - 50 < tr["cx"] < width + 80
        ]
        return self._merge_center_candidates(output)

    def _stabilize_sites(self, sites: list[dict], width: int, height: int) -> list[dict]:
        side_indices = [idx for idx, site in enumerate(sites) if site["kind"].startswith("side")]
        output = [dict(site) for site in sites if not site["kind"].startswith("side")]
        unmatched_tracks = set(range(len(self._site_tracks)))
        assignments: list[tuple[int, int]] = []

        for site_idx in side_indices:
            site = sites[site_idx]
            sx, sy = site["center"]
            best_track = None
            best_dist = float("inf")
            for tr_idx in unmatched_tracks:
                tr = self._site_tracks[tr_idx]
                if tr["kind"] != site["kind"]:
                    continue
                pred_x = tr["cx"] + float(np.clip(tr.get("vx", 0.0), -24.0, 24.0))
                pred_y = tr["cy"] + float(np.clip(tr.get("vy", 0.0), -10.0, 10.0))
                dist = float(np.hypot(sx - pred_x, sy - pred_y))
                gate = 92.0 if site.get("direct_side") else 76.0
                if dist < gate and dist < best_dist:
                    best_track = tr_idx
                    best_dist = dist
            if best_track is not None:
                unmatched_tracks.remove(best_track)
                assignments.append((best_track, site_idx))

        matched_sites = {site_idx for _, site_idx in assignments}
        for tr_idx, site_idx in assignments:
            tr = self._site_tracks[tr_idx]
            site = dict(sites[site_idx])
            raw_x, raw_y = site["center"]
            raw_w, raw_h = site["box_xywh"][2], site["box_xywh"][3]
            old_x, old_y = tr["cx"], tr["cy"]
            jump = float(np.hypot(raw_x - old_x, raw_y - old_y))
            if jump <= 7.0:
                alpha = 0.18 if site.get("direct_side") else 0.12
            elif jump <= 64.0:
                alpha = 0.48 if site.get("direct_side") else 0.24
            else:
                alpha = 0.26 if site.get("direct_side") else (0.10 if tr["age"] >= 4 else 0.24)
            tr["cx"] = (1.0 - alpha) * old_x + alpha * raw_x
            tr["cy"] = (1.0 - alpha) * old_y + alpha * raw_y
            mvx = tr["cx"] - old_x
            mvy = tr["cy"] - old_y
            old_vx = tr.get("vx", 0.0)
            old_vy = tr.get("vy", 0.0)
            tr["vx"] = 0.66 * old_vx + 0.34 * mvx
            tr["vy"] = 0.72 * old_vy + 0.28 * mvy
            if mvx * old_vx < 0 or abs(mvx) < 0.6:
                tr["vx"] = 0.0
            if mvy * old_vy < 0 or abs(mvy) < 0.6:
                tr["vy"] = 0.0
            tr["w"] = 0.985 * tr["w"] + 0.015 * raw_w
            tr["h"] = 0.985 * tr["h"] + 0.015 * raw_h
            tr["last_site"] = site
            tr["missed"] = 0
            tr["age"] += 1
            output.append(self._site_from_track(tr, site, width, height, direct=bool(site.get("direct_side"))))

        for site_idx in side_indices:
            if site_idx in matched_sites:
                continue
            site = dict(sites[site_idx])
            sx, sy = site["center"]
            box_w, box_h = site["box_xywh"][2], site["box_xywh"][3]
            tr = {
                "id": self._next_site_id,
                "kind": site["kind"],
                "cx": float(sx),
                "cy": float(sy),
                "vx": 0.0,
                "vy": 0.0,
                "w": float(box_w),
                "h": float(box_h),
                "last_site": site,
                "missed": 0,
                "age": 1,
            }
            self._next_site_id += 1
            self._site_tracks.append(tr)
            output.append(site)

        for tr_idx in list(unmatched_tracks):
            tr = self._site_tracks[tr_idx]
            tr["vx"] = 0.25 * tr.get("vx", 0.0)
            tr["vy"] = 0.25 * tr.get("vy", 0.0)
            tr["missed"] += 1
            if (
                tr["missed"] <= self.side_missing_grace
                and tr["age"] >= 4
                and self.right_x0 < tr["cx"] < width - 8
                and 250 < tr["cy"] < height - 40
            ):
                output.append(self._site_from_track(tr, tr["last_site"], width, height, direct=True))

        self._site_tracks = [
            tr
            for tr in self._site_tracks
            if tr["missed"] <= self.side_missing_grace + 2 and self.right_x0 - 60 < tr["cx"] < width + 90 and 220 < tr["cy"] < height + 50
        ]
        return sorted(output, key=lambda item: (item.get("tube_index", 999), item["kind"], item["center"][1], item["center"][0]))

    def _site_from_track(self, tr: dict, template: dict, width: int, height: int, direct: bool) -> dict:
        cx = float(tr["cx"])
        cy = float(tr["cy"])
        bw = int(round(tr["w"]))
        bh = int(round(tr["h"]))
        x1, y1, x2, y2 = _safe_rect(int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2), width, height)
        complete = x1 >= self.right_x0 and (x2 - x1) >= int(bw * 0.72) and (y2 - y1) >= int(bh * 0.72)
        site = dict(template)
        site["center"] = (cx, cy)
        site["box"] = (x1, y1, x2, y2)
        site["box_xywh"] = (x1, y1, x2 - x1, y2 - y1)
        site["complete"] = complete
        site["center_source"] = str(template.get("center_source", "")) + "+site_track"
        if direct:
            site["direct_side"] = True
        else:
            site.pop("direct_side", None)
            site["tracked_side"] = True
        site["site_track_id"] = tr["id"]
        return site

    def _expected_sites(self, frame: np.ndarray, use_history: bool = False) -> list[dict]:
        h, w = frame.shape[:2]
        sites: list[dict] = []
        centers = self._infer_container_centers(frame)
        if use_history:
            centers = self._stabilize_centers(centers, w)
        for tube_idx, center in enumerate(centers):
            cx = center["cx"]
            tube_box = center["tube_box"]
            if cx >= self.side_lower_far_x_min:
                lower_offset = self.side_lower_far_offset
                lower_y = self.side_lower_far_y
                lower_hw, lower_hh = self.side_lower_far_half
            else:
                lower_offset = self.side_lower_offset
                lower_y = self.side_lower_y
                lower_hw, lower_hh = 34, 34
            definitions = [
                ("top", cx, 318, 66, 62),
                ("bottom", cx, 962, 70, 72),
                ("side_upper", cx + self.side_upper_offset, self.side_upper_y, 34, 36),
                ("side_lower", cx + lower_offset, lower_y, lower_hw, lower_hh),
            ]
            for kind, sx, sy, hw, hh in definitions:
                side_source = ""
                if kind.startswith("side"):
                    side_source = "+red_ref_locked"
                x1, y1, x2, y2 = _safe_rect(int(sx - hw), int(sy - hh), int(sx + hw), int(sy + hh), w, h)
                complete = (x2 - x1) >= int(hw * 1.45) and (y2 - y1) >= int(hh * 1.45) and x1 >= self.right_x0
                sites.append(
                    {
                        "kind": kind,
                        "tube_index": tube_idx,
                        "center": (float(sx), float(sy)),
                        "box": (x1, y1, x2, y2),
                        "box_xywh": (x1, y1, x2 - x1, y2 - y1),
                        "complete": complete,
                        "tube_box": tube_box,
                        "center_source": center["source"] + side_source,
                        "geometry_locked": bool(kind.startswith("side")),
                        "local_side": False,
                        "side_quality": 0.0,
                    }
                )
        return sites

    def _match_side_anchor(
        self,
        kind: str,
        sx: float,
        sy: float,
        container_cx: float,
        side_caps: list[dict],
        used_side: set[int],
    ) -> int | None:
        best_idx = None
        best_score = float("inf")
        for idx, cap in enumerate(side_caps):
            if idx in used_side or cap["kind"] != kind:
                continue
            cap_x, cap_y = cap["center"]
            if not self._side_cap_matches_container(kind, cap_x, cap_y, container_cx, sx, sy):
                continue
            dx = abs(cap_x - sx)
            dy = abs(cap_y - sy)
            score = dx / 42.0 + dy / 24.0 - 0.10 * float(cap.get("quality", 0.0))
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _side_cap_matches_container(self, kind: str, cap_x: float, cap_y: float, container_cx: float, sx: float, sy: float) -> bool:
        offset = container_cx - cap_x
        if not 55.0 <= offset <= 140.0:
            return False
        if kind == "side_upper":
            return 385.0 <= cap_y <= 435.0 and abs(cap_x - sx) <= 58.0 and abs(cap_y - sy) <= 34.0
        return 815.0 <= cap_y <= 872.0 and abs(cap_x - sx) <= 62.0 and abs(cap_y - sy) <= 36.0

    def _blend_side_center(self, kind: str, sx: float, sy: float, direct_center: tuple[float, float]) -> tuple[float, float]:
        cap_x, cap_y = direct_center
        if kind == "side_lower":
            return 0.18 * sx + 0.82 * cap_x, 0.10 * sy + 0.90 * cap_y
        return 0.25 * sx + 0.75 * cap_x, 0.22 * sy + 0.78 * cap_y

    def _site_feature(self, frame: np.ndarray, site: dict) -> tuple[np.ndarray, dict]:
        x1, y1, x2, y2 = site["box"]
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or not site["complete"]:
            return np.zeros(16, dtype=np.float32), {"skip": True, "reason": "incomplete"}
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin = (((h <= 25) | (h >= 165)) & (s >= 38) & (v >= 55)).mean()
        purple = ((h >= 135) & (h <= 172) & (s >= 65) & (v >= 45)).mean()
        blue = cv2.inRange(hsv, (90, 45, 30), (135, 255, 255))
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        gray_mask = (((s < 72) & (v > 45) & (v < 245))).astype(np.uint8) * 255
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        area = float(max(patch.shape[0] * patch.shape[1], 1))
        blue_area = float(cv2.countNonZero(blue))
        gray_area = float(cv2.countNonZero(gray_mask))
        edge_mean = float(edge.mean())
        value_std = float(v.std())
        gray_ratio = gray_area / area
        edge_score = float(np.clip((edge_mean - 8.0) / 18.0, 0.0, 1.0))
        texture_score = float(np.clip((value_std - 18.0) / 38.0, 0.0, 1.0))
        gray_score = float(np.clip(gray_ratio / 0.72, 0.0, 1.0))
        side_presence = 0.55 * edge_score + 0.25 * texture_score + 0.20 * gray_score
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        hist_h = hist_h / max(float(hist_h.sum()), 1.0)
        feat = np.array(
            [
                blue_area / area,
                gray_area / area,
                float(s.mean()) / 255.0,
                float(s.std()) / 255.0,
                float(v.mean()) / 255.0,
                value_std / 255.0,
                edge_mean / 255.0,
                float((v < 55).mean()),
                float((v > 190).mean()),
                float(skin),
                float(purple),
                *hist_h[:5].tolist(),
            ],
            dtype=np.float32,
        )
        direct_side = bool(site.get("direct_side"))
        skin_limit = 0.68 if direct_side else 0.46
        purple_limit = 0.72 if direct_side else 0.58
        skip = self.skip_occluded and (skin > skin_limit or purple > purple_limit)
        meta = {
            "skip": bool(skip),
            "reason": "occluded" if skip else "",
            "blue_area": blue_area,
            "gray_area": gray_area,
            "gray_ratio": float(gray_ratio),
            "edge_mean": edge_mean,
            "side_presence": float(side_presence),
            "skin_ratio": float(skin),
            "purple_ratio": float(purple),
            "sat_mean": float(s.mean()),
            "value_mean": float(v.mean()),
            "value_std": value_std,
        }
        return feat, meta


def save_model(detector: VideoADetector | VideoBDetector, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(detector, f)


def load_model(path: str | Path) -> VideoADetector | VideoBDetector:
    with Path(path).open("rb") as f:
        return pickle.load(f)
