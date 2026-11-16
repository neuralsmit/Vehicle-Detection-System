"""
Vehicle Detection and Counting using YOLOv8 + OpenCV + ByteTrack.

Usage:
    python vehicle_counter.py --source input_videos/traffic1.mp4 --output outputs/traffic1_out.mp4
    python vehicle_counter.py --source 0 --show
"""

import os
# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any

import cv2
from ultralytics import YOLO

# COCO class IDs mapped to standard categories
VEHICLE_CLASSES = {
    2: "Cars",
    3: "Motorcycles",
    5: "Buses",
    7: "Trucks",
}


def ensure_model_exists(model_path: str = "yolov8n.pt"):
    """Restore model from backup if corrupted or missing to prevent offline failures."""
    if not os.path.exists(model_path) and os.path.exists("yolov8n_backup.pt"):
        shutil.copyfile("yolov8n_backup.pt", model_path)


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]
) -> bool:
    """Return True if line segment AB intersects line segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def point_side(P: Tuple[int, int], A: Tuple[int, int], B: Tuple[int, int]) -> float:
    """Calculate signed cross product to determine which side of line AB point P is on."""
    return (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])


def resolve_counting_line(
    line_config: Any, width: int, height: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Resolve flexible line configuration into concrete endpoint coordinates: (p1, p2).
    """
    if isinstance(line_config, (int, float)):
        y = int(height * float(line_config))
        return (0, y), (width, y)
    elif isinstance(line_config, dict):
        if "p1" in line_config and "p2" in line_config:
            return tuple(line_config["p1"]), tuple(line_config["p2"])
        orientation = line_config.get("orientation", "horizontal").lower()
        pos = line_config.get("pos", 0.5)
        if orientation == "vertical":
            x = int(width * pos)
            ymin = int(line_config.get("ymin", 0))
            ymax = int(line_config.get("ymax", height))
            return (x, ymin), (x, ymax)
        else:
            y = int(height * pos)
            xmin = int(line_config.get("xmin", 0))
            xmax = int(line_config.get("xmax", width))
            return (xmin, y), (xmax, y)
    y = int(height * 0.5)
    return (0, y), (width, y)


def process_video(
    source: str,
    output_path: Optional[str] = None,
    model_path: str = "yolov8n.pt",
    line_config: Any = 0.5,
    conf_threshold: float = 0.25,
    max_dim: Optional[int] = 1280,
    show_preview: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Process a single video for vehicle detection, ByteTrack tracking, and line-crossing counting.
    Automatically scales high-res (e.g. 4K) videos to max_dim for efficient processing.
    """
    ensure_model_exists(model_path)
    src_display = str(source)
    is_cam = str(source).isdigit()
    source_val = int(source) if is_cam else source

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else -1
    duration_sec = total_frames / fps_in if total_frames > 0 else 0.0

    # Determine processing dimensions (scale down 4K to max_dim for speed and reasonable output size)
    if max_dim and max(raw_width, raw_height) > max_dim:
        scale = max_dim / max(raw_width, raw_height)
        width = int(raw_width * scale)
        height = int(raw_height * scale)
    else:
        scale = 1.0
        width = raw_width
        height = raw_height

    p1, p2 = resolve_counting_line(line_config, width, height)

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    model = YOLO(model_path)

    last_centers: Dict[int, Tuple[int, int]] = {}
    counted_ids = set()
    counted_dir_a = set()
    counted_dir_b = set()
    id_to_class: Dict[int, str] = {}
    counts = {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0}

    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t_frame_start = time.perf_counter()

            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                conf=conf_threshold,
                verbose=False,
            )

            result = results[0] if results else None
            boxes = result.boxes if result is not None else None

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()

                for box, tid, cid in zip(xyxy, track_ids, class_ids):
                    category = VEHICLE_CLASSES.get(cid, "Cars")
                    id_to_class[tid] = category

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    curr_pt = (cx, cy)
                    prev_pt = last_centers.get(tid)

                    if prev_pt is not None and tid not in counted_ids:
                        if segments_intersect(prev_pt, curr_pt, p1, p2):
                            counted_ids.add(tid)
                            counts[id_to_class[tid]] += 1
                            s = point_side(curr_pt, p1, p2)
                            if s >= 0:
                                counted_dir_a.add(tid)
                            else:
                                counted_dir_b.add(tid)

                    last_centers[tid] = curr_pt

                    # Annotations
                    is_counted = tid in counted_ids
                    box_color = (0, 220, 0) if is_counted else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label_text = f"#{tid} {category}"
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )
                    cv2.circle(frame, curr_pt, 4, (0, 0, 255), -1)

            # Draw counting line
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            mid_line = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                frame,
                "COUNTING LINE",
                (max(mid_line[0] - 80, 10), max(mid_line[1] - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            # Performance stats
            t_now = time.perf_counter()
            frame_elapsed = max(t_now - t_frame_start, 1e-6)
            instant_fps = 1.0 / frame_elapsed
            total_elapsed = max(t_now - t_start, 1e-6)
            avg_fps = frame_idx / total_elapsed

            # Overlay HUD Dashboard
            overlay = frame.copy()
            hud_w = min(420, width - 20)
            cv2.rectangle(overlay, (10, 10), (10 + hud_w, 165), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            hud_lines = [
                f"Source: {os.path.basename(src_display)} ({width}x{height})",
                f"Frame: {frame_idx}/{total_frames if total_frames > 0 else 'Live'}  FPS: {instant_fps:.1f} (Avg: {avg_fps:.1f})",
                f"Total Counted: {len(counted_ids)} (A: {len(counted_dir_a)}, B: {len(counted_dir_b)})",
                f"Cars: {counts['Cars']}   Motorcycles: {counts['Motorcycles']}",
                f"Buses: {counts['Buses']}   Trucks: {counts['Trucks']}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    frame,
                    line,
                    (20, 32 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )

            if writer:
                writer.write(frame)

            if show_preview:
                cv2.imshow("Vehicle Detection & Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if progress_callback and frame_idx % 20 == 0:
                progress_callback(frame_idx, total_frames, avg_fps)

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_time = max(time.perf_counter() - t_start, 1e-6)
    final_avg_fps = round(frame_idx / total_time, 2)
    total_vehicles = len(counted_ids)

    return {
        "video": os.path.basename(src_display),
        "source_path": src_display,
        "output_path": output_path,
        "raw_resolution": f"{raw_width}x{raw_height}",
        "resolution": f"{width}x{height}",
        "total_frames": frame_idx,
        "video_duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(total_time, 2),
        "avg_fps": final_avg_fps,
        "total_vehicles": total_vehicles,
        "counts": counts,
        "dir_a_count": len(counted_dir_a),
        "dir_b_count": len(counted_dir_b),
        "status": "Success",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle detection and counting using YOLOv8 + ByteTrack")
    p.add_argument("--source", default="input_videos/traffic1.mp4", help="Video path or camera index")
    p.add_argument("--output", default="outputs/output.mp4", help="Path of the annotated output video")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path")
    p.add_argument("--line", type=float, default=0.55, help="Counting line position (0.0 to 1.0 of frame height)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--max-dim", type=int, default=1280, help="Max dimension to scale frames for performance")
    p.add_argument("--show", action="store_true", help="Display live preview window")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Processing single video: {args.source}")
    try:
        stats = process_video(
            source=args.source,
            output_path=args.output,
            model_path=args.model,
            line_config=args.line,
            conf_threshold=args.conf,
            max_dim=args.max_dim,
            show_preview=args.show,
        )
        print("\n" + "=" * 50)
        print("Processing Completed Successfully:")
        print(f"  Video:           {stats['video']}")
        print(f"  Resolution:      {stats['resolution']} (Raw: {stats['raw_resolution']})")
        print(f"  Processed Frames:{stats['total_frames']}")
        print(f"  Avg FPS:         {stats['avg_fps']}")
        print(f"  Total Vehicles:  {stats['total_vehicles']}")
        print(f"  Counts Breakdown:{stats['counts']}")
        print(f"  Output Saved to: {stats['output_path']}")
        print("=" * 50)
    except Exception as e:
        print(f"Error processing video {args.source}: {e}")


if __name__ == "__main__":
    main()
