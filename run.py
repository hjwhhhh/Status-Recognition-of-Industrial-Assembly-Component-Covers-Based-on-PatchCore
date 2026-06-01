from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.detectors import CYAN, GREEN, RED, WHITE, YELLOW, VideoADetector, VideoBDetector, load_model, save_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


CONFIGS = {
    "A": {
        "video": ROOT / "videoA.mp4",
        "model": MODEL_DIR / "videoA_detector.pkl",
        "normal_end_sec": 360.0,
        "train_stride_sec": 1.0,
        "demo_start_sec": 370.0,
        "demo_duration_sec": 25.0,
        "detector": VideoADetector,
    },
    "B": {
        "video": ROOT / "videoB.mp4",
        "model": MODEL_DIR / "videoB_detector.pkl",
        "normal_end_sec": 180.0,
        "train_stride_sec": 1.0,
        "demo_start_sec": 980.0,
        "demo_duration_sec": 25.0,
        "detector": VideoBDetector,
    },
}


def train_one(video_key: str) -> None:
    cfg = CONFIGS[video_key]
    detector = cfg["detector"]()
    print(f"[train] video{video_key}: {cfg['video'].name}")
    info = detector.train(
        cfg["video"],
        normal_end_sec=cfg["normal_end_sec"],
        stride_sec=cfg["train_stride_sec"],
    )
    save_model(detector, cfg["model"])
    print(f"[train] saved {cfg['model']}")
    print(f"[train] info: {info}")


def ensure_model(video_key: str):
    cfg = CONFIGS[video_key]
    model_path = cfg["model"]
    expected_version = getattr(cfg["detector"](), "version", None)
    if not model_path.exists():
        train_one(video_key)
    detector = load_model(model_path)
    missing_required = False
    if video_key == "B":
        required_attrs = (
            "model_signature",
            "side_upper_offset",
            "side_lower_offset",
            "side_upper_y",
            "side_lower_y",
            "side_lower_far_x_min",
            "side_lower_far_offset",
            "side_lower_far_y",
            "side_lower_far_half",
            "side_present_min",
            "side_gray_ng_min",
            "side_edge_ng_min",
            "side_removed_edge_min",
            "side_removed_value_std_min",
        )
        for attr in required_attrs:
            if getattr(detector, attr, None) is None:
                missing_required = True
                break
        if (
            getattr(detector, "model_signature", None) != "videoB_removed_side_ng_v21"
            or
            getattr(detector, "side_upper_offset", None) != -95
            or getattr(detector, "side_lower_offset", None) != -76
            or getattr(detector, "side_upper_y", None) != 415
            or getattr(detector, "side_lower_y", None) != 870
            or getattr(detector, "side_lower_far_x_min", None) != 1760.0
            or getattr(detector, "side_lower_far_offset", None) != -116
            or getattr(detector, "side_lower_far_y", None) != 888
            or getattr(detector, "side_lower_far_half", None) != (44, 44)
            or getattr(detector, "side_removed_edge_min", None) != 88.0
            or getattr(detector, "side_removed_value_std_min", None) != 43.0
        ):
            missing_required = True
    if (expected_version is not None and getattr(detector, "version", None) != expected_version) or missing_required:
        print(
            f"[train] video{video_key}: stale model version "
            f"{getattr(detector, 'version', None)} -> {expected_version}"
        )
        train_one(video_key)
        detector = load_model(model_path)
    return detector


def draw_original_label(frame: np.ndarray, text: str, sec: float) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, "Original", (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3, cv2.LINE_AA)
    cv2.putText(out, f"{text}  t={sec:.1f}s", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    return out


def resize_view(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    height = int(round(h * width / w))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def summarize_metrics(video_key: str, detector, detections) -> dict[str, float]:
    valid = [d for d in detections if d.status != "SKIP"]
    ng = [d for d in valid if d.status == "NG"]
    summary: dict[str, float] = {
        "status": 1.0 if ng else 0.0,
        "valid": float(len(valid)),
        "ng": float(len(ng)),
        "score": float(max([d.score for d in valid], default=0.0)),
    }
    if video_key == "A":
        summary["yellow"] = float(np.mean([d.details.get("yellow_ratio", 0.0) for d in valid])) if valid else 0.0
        summary["sat"] = float(np.mean([d.details.get("sat_mean", 0.0) for d in valid])) if valid else 0.0
        summary["texture"] = float(np.mean([d.details.get("texture", 0.0) for d in valid])) if valid else 0.0
        summary["yellow_min"] = float(getattr(detector, "yellow_min", 0.0))
        summary["ok_sat_min"] = float(getattr(detector, "ok_sat_min", 0.0))
    else:
        main = [d for d in valid if d.details.get("group") in ("top", "bottom")]
        side = [d for d in valid if d.details.get("group") == "side"]
        capped_side = [d for d in side if d.status == "OK"]
        open_side = [d for d in side if d.status == "NG"]
        summary["main_blue"] = float(np.mean([d.details.get("blue_area", 0.0) for d in main])) if main else 0.0
        summary["side_gray"] = float(np.mean([d.details.get("gray_area", 0.0) for d in side])) if side else 0.0
        summary["side_edge"] = float(np.mean([d.details.get("edge_mean", 0.0) for d in side])) if side else 0.0
        summary["side_presence"] = float(np.mean([d.details.get("side_presence", 0.0) for d in side])) if side else 0.0
        summary["side_capped"] = float(len(capped_side))
        summary["side_open"] = float(len(open_side))
        summary["side_removed"] = float(len(open_side))
        summary["side_target"] = float(len(side))
        summary["top_blue_min"] = float(getattr(detector, "blue_min", {}).get("top", 0.0))
        summary["bottom_blue_min"] = float(getattr(detector, "blue_min", {}).get("bottom", 0.0))
        summary["side_gray_min"] = float(getattr(detector, "side_gray_ng_min", getattr(detector, "gray_min", 0.0)))
        summary["side_edge_min"] = float(getattr(detector, "side_edge_ng_min", 0.0))
        summary["side_present_min"] = float(getattr(detector, "side_present_min", 0.0))
    return summary


def draw_metric_plot(
    panel: np.ndarray,
    history: list[dict[str, float]],
    key: str,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    scale_max: float | None = None,
    threshold: float | None = None,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (90, 90, 90), 1)
    if len(history) < 2:
        return
    vals = np.array([item.get(key, 0.0) for item in history], dtype=np.float32)
    if scale_max is None:
        vmax = float(max(np.percentile(vals, 95), vals.max(), 1e-6))
    else:
        vmax = max(float(scale_max), 1e-6)
    vals = np.clip(vals / vmax, 0.0, 1.0)
    if threshold is not None:
        ty = int(y + h - 3 - np.clip(threshold / vmax, 0.0, 1.0) * (h - 6))
        cv2.line(panel, (x + 2, ty), (x + w - 2, ty), (210, 210, 210), 1, cv2.LINE_AA)
    xs = np.linspace(x + 2, x + w - 2, len(vals)).astype(np.int32)
    ys = (y + h - 3 - vals * (h - 6)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)


def draw_parameter_panel(
    frame: np.ndarray,
    video_key: str,
    detector,
    detections,
    sec: float,
    history: deque[dict[str, float]],
) -> np.ndarray:
    metrics = summarize_metrics(video_key, detector, detections)
    metrics["time"] = float(sec)
    history.append(metrics)

    out = frame.copy()
    h, w = out.shape[:2]
    panel_w, panel_h = 650, 250
    x0, y0 = 24, h - panel_h - 24
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0.0, out)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (230, 230, 230), 1)

    title = f"video{video_key} realtime parameters  t={sec:.1f}s"
    cv2.putText(out, title, (x0 + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, WHITE, 2, cv2.LINE_AA)
    if video_key == "A":
        lines = [
            f"status={'NG' if metrics['status'] > 0 else 'OK'}  faces={metrics['valid']:.0f}  max_score={metrics['score']:.2f}",
            f"yellow={metrics['yellow']:.2f}  yellow_min={metrics['yellow_min']:.2f}",
            f"sat={metrics['sat']:.0f}  ok_sat_min={metrics['ok_sat_min']:.0f}  texture={metrics['texture']:.0f}",
        ]
        plot_specs = [("yellow", 1.0, GREEN), ("sat", 255.0, CYAN), ("texture", 220.0, YELLOW)]
    else:
        lines = [
            f"status={'NG' if metrics['status'] > 0 else 'OK'}  sites={metrics['valid']:.0f}  NG={metrics['ng']:.0f}",
            f"main_blue={metrics['main_blue']:.0f}  top_min={metrics['top_blue_min']:.0f}  bottom_min={metrics['bottom_blue_min']:.0f}",
            f"side_presence={metrics['side_presence']:.2f}  threshold={metrics['side_present_min']:.2f}  capped/removed={metrics['side_capped']:.0f}/{metrics['side_removed']:.0f}",
        ]
        plot_specs = [
            ("side_presence", 1.0, GREEN, metrics["side_present_min"]),
            ("side_removed", 6.0, RED, None),
            ("main_blue", 6500.0, CYAN, None),
        ]

    for idx, line in enumerate(lines):
        cv2.putText(out, line, (x0 + 16, y0 + 62 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 2, cv2.LINE_AA)

    hist = list(history)
    for idx, spec in enumerate(plot_specs):
        key, scale_max, color, threshold = spec if len(spec) == 4 else (*spec, None)
        py = y0 + 150 + idx * 28
        cv2.putText(out, key, (x0 + 16, py + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        draw_metric_plot(out, hist, key, (x0 + 108, py, panel_w - 128, 22), color, scale_max, threshold)
    return out


def make_demo(video_key: str, start: float | None, duration: float | None, out_path: Path | None, width_per_view: int) -> Path:
    cfg = CONFIGS[video_key]
    detector = ensure_model(video_key)
    video_path = cfg["video"]
    start = cfg["demo_start_sec"] if start is None else start
    duration = cfg["demo_duration_sec"] if duration is None else duration
    out_path = out_path or (OUTPUT_DIR / f"demo_video{video_key}_{int(start)}s_{int(duration)}s.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(round(duration * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    view_h = int(round(1080 * width_per_view / 1920))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width_per_view * 2, view_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")

    print(f"[demo] video{video_key}: {start:.1f}s - {start + duration:.1f}s -> {out_path}")
    if hasattr(detector, "reset_temporal"):
        detector.reset_temporal()
    predict_frame = getattr(detector, "predict_stable", detector.predict)
    metric_history: deque[dict[str, float]] = deque(maxlen=180)
    ng_frames = 0
    processed = 0
    for i in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        sec = start + i / fps
        detections = predict_frame(frame)
        if any(det.status == "NG" for det in detections):
            ng_frames += 1
        original = draw_original_label(frame, f"video{video_key}", sec)
        annotated = detector.draw(frame, detections)
        annotated = draw_parameter_panel(annotated, video_key, detector, detections, sec, metric_history)
        left = resize_view(original, width_per_view)
        right = resize_view(annotated, width_per_view)
        side_by_side = np.hstack([left, right])
        writer.write(side_by_side)
        processed += 1
    cap.release()
    writer.release()
    print(f"[demo] frames={processed}, ng_frames={ng_frames}")
    return out_path


def infer_csv(video_key: str, start: float, duration: float, stride_sec: float, out_path: Path) -> Path:
    cfg = CONFIGS[video_key]
    detector = ensure_model(video_key)
    cap = cv2.VideoCapture(str(cfg["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    steps = int(duration / stride_sec) + 1
    for i in range(steps):
        sec = start + i * stride_sec
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.predict(frame)
        frame_status = "NG" if any(d.status == "NG" for d in detections) else ("OK" if detections else "WAIT")
        rows.append(
            {
                "video": f"video{video_key}",
                "time_sec": f"{sec:.3f}",
                "frame_status": frame_status,
                "detections": len(detections),
                "ng_detections": sum(1 for d in detections if d.status == "NG"),
                "max_score": f"{max([d.score for d in detections], default=0.0):.6f}",
            }
        )
    cap.release()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["video", "time_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[infer] wrote {out_path}, rows={len(rows)}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normal-only industrial cap inspection for videoA/videoB.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train normal-only detector")
    train.add_argument("--video", choices=["A", "B", "all"], default="all")

    demo = sub.add_parser("demo", help="create side-by-side demo video")
    demo.add_argument("--video", choices=["A", "B", "all"], default="all")
    demo.add_argument("--start", type=float, default=None)
    demo.add_argument("--duration", type=float, default=None)
    demo.add_argument("--out", type=Path, default=None)
    demo.add_argument("--width-per-view", type=int, default=960)

    infer = sub.add_parser("infer", help="write low-rate frame-level CSV")
    infer.add_argument("--video", choices=["A", "B"], required=True)
    infer.add_argument("--start", type=float, required=True)
    infer.add_argument("--duration", type=float, required=True)
    infer.add_argument("--stride-sec", type=float, default=1.0)
    infer.add_argument("--out", type=Path, default=None)

    all_cmd = sub.add_parser("all", help="train both detectors and create default demos")
    all_cmd.add_argument("--width-per-view", type=int, default=960)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        keys = ["A", "B"] if args.video == "all" else [args.video]
        for key in keys:
            train_one(key)
    elif args.command == "demo":
        keys = ["A", "B"] if args.video == "all" else [args.video]
        for key in keys:
            if args.out is not None and len(keys) > 1:
                out_path = args.out.with_name(f"{args.out.stem}_video{key}{args.out.suffix}")
            else:
                out_path = args.out
            make_demo(key, args.start, args.duration, out_path, args.width_per_view)
    elif args.command == "infer":
        out_path = args.out or (OUTPUT_DIR / f"infer_video{args.video}_{int(args.start)}s.csv")
        infer_csv(args.video, args.start, args.duration, args.stride_sec, out_path)
    elif args.command == "all":
        for key in ["A", "B"]:
            train_one(key)
            make_demo(key, None, None, None, args.width_per_view)


if __name__ == "__main__":
    main()
