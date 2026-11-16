"""
Batch Video Processing Script for Vehicle Detection, Tracking, and Counting.

Uses YOLOv8 + ByteTrack to detect, track, and count vehicles crossing defined
lines across all video feeds in input_videos/, generating separate output videos
and a comparative CSV report in results/.

Usage:
    python batch_process.py
    python batch_process.py --input-dir input_videos --output-dir outputs --results-dir results
"""

import os
# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import time
from typing import Dict, List, Any

from vehicle_counter import process_video

# Default per-video counting line setups tailored to camera geometry
VIDEO_LINE_CONFIGS: Dict[str, Any] = {
    # New Traffic Videos
    "traffic1.mp4": {
        "orientation": "horizontal",
        "pos": 0.55,
    },
    "traffic2.mp4": {
        "orientation": "horizontal",
        "pos": 0.65,
    },
    "traffic3.mp4": {
        "orientation": "vertical",
        "pos": 0.50,
    },
    
}

DEFAULT_LINE_CONFIG = {"orientation": "horizontal", "pos": 0.50}


def parse_args():
    p = argparse.ArgumentParser(description="Batch Vehicle Detection and Counting")
    p.add_argument("--input-dir", default="input_videos", help="Directory containing source videos")
    p.add_argument("--output-dir", default="outputs", help="Directory to save annotated videos")
    p.add_argument("--results-dir", default="results", help="Directory to save CSV/JSON reports")
    p.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 weights")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--max-dim", type=int, default=1280, help="Max dimension to scale frames for performance")
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Specific video filenames to process (defaults to all videos found in input_videos/)",
    )
    return p.parse_args()


def save_csv_report(results: List[Dict[str, Any]], csv_path: str):
    """Write comprehensive comparison metrics to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "Video",
        "Status",
        "Raw_Resolution",
        "Processed_Resolution",
        "Total_Frames",
        "Video_Duration_Sec",
        "Processing_Time_Sec",
        "Avg_FPS",
        "Total_Vehicles",
        "Cars",
        "Motorcycles",
        "Buses",
        "Trucks",
        "Output_File",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            counts = r.get("counts", {})
            writer.writerow({
                "Video": r.get("video", "N/A"),
                "Status": r.get("status", "Unknown"),
                "Raw_Resolution": r.get("raw_resolution", "N/A"),
                "Processed_Resolution": r.get("resolution", "N/A"),
                "Total_Frames": r.get("total_frames", 0),
                "Video_Duration_Sec": r.get("video_duration_sec", 0.0),
                "Processing_Time_Sec": r.get("processing_time_sec", 0.0),
                "Avg_FPS": r.get("avg_fps", 0.0),
                "Total_Vehicles": r.get("total_vehicles", 0),
                "Cars": counts.get("Cars", 0),
                "Motorcycles": counts.get("Motorcycles", 0),
                "Buses": counts.get("Buses", 0),
                "Trucks": counts.get("Trucks", 0),
                "Output_File": os.path.basename(r.get("output_path", "")) if r.get("output_path") else "N/A",
            })


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print an ASCII comparison table of all processed videos."""
    header = f"{'Video':<16} | {'Status':<8} | {'Frames':<7} | {'Dur(s)':<7} | {'Avg FPS':<8} | {'Total':<6} | {'Cars':<5} | {'Motos':<6} | {'Buses':<6} | {'Trucks':<6}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("                     BATCH PROCESSING COMPARISON REPORT")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        counts = r.get("counts", {})
        vid = r.get("video", "N/A")
        status = r.get("status", "N/A")
        frames = str(r.get("total_frames", 0))
        dur = f"{r.get('video_duration_sec', 0.0):.1f}"
        fps = f"{r.get('avg_fps', 0.0):.1f}"
        tot = str(r.get("total_vehicles", 0))
        cars = str(counts.get("Cars", 0))
        motos = str(counts.get("Motorcycles", 0))
        buses = str(counts.get("Buses", 0))
        trucks = str(counts.get("Trucks", 0))
        print(f"{vid:<16} | {status:<8} | {frames:<7} | {dur:<7} | {fps:<8} | {tot:<6} | {cars:<5} | {motos:<6} | {buses:<6} | {trucks:<6}")
    print(sep)


def run_batch():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    results_dir = args.results_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Automatically discover video files if none specified
    if args.videos:
        target_videos = args.videos
    else:
        valid_exts = (".mp4", ".avi", ".mov", ".mkv")
        target_videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    print("=" * 65)
    print("       VEHICLE DETECTION & COUNTING - BATCH PROCESSOR")
    print(f"  Input Directory:   {input_dir}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Results Directory: {results_dir}")
    print(f"  YOLO Model:        {args.model}")
    print(f"  Confidence:        {args.conf}")
    print(f"  Target Videos:     {', '.join(target_videos)}")
    print("=" * 65)

    all_results = []
    t_batch_start = time.perf_counter()

    for idx, video_name in enumerate(target_videos, start=1):
        input_path = os.path.join(input_dir, video_name)
        out_base = os.path.splitext(video_name)[0] + "_counted.mp4"
        output_path = os.path.join(output_dir, out_base)

        print(f"\n[{idx}/{len(target_videos)}] Processing: {video_name} ...")

        if not os.path.exists(input_path):
            print(f"  [ERROR] Input video file not found: {input_path}")
            all_results.append({
                "video": video_name,
                "status": "Failed (File Not Found)",
                "total_frames": 0,
                "video_duration_sec": 0.0,
                "processing_time_sec": 0.0,
                "avg_fps": 0.0,
                "total_vehicles": 0,
                "counts": {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0},
                "output_path": "",
            })
            continue

        line_cfg = VIDEO_LINE_CONFIGS.get(video_name, DEFAULT_LINE_CONFIG)

        def make_progress_logger(vname):
            def log_progress(cur, tot, cur_fps):
                pct = (cur / tot * 100) if tot > 0 else 0
                print(f"    -> [{vname}] Frame {cur}/{tot} ({pct:.1f}%) | FPS: {cur_fps:.1f}", end="\r", flush=True)
            return log_progress

        try:
            stats = process_video(
                source=input_path,
                output_path=output_path,
                model_path=args.model,
                line_config=line_cfg,
                conf_threshold=args.conf,
                max_dim=args.max_dim,
                show_preview=False,
                progress_callback=make_progress_logger(video_name),
            )
            print()  # clear carriage return line
            print(f"  [OK] Completed in {stats['processing_time_sec']}s (Avg FPS: {stats['avg_fps']})")
            print(f"       Counted {stats['total_vehicles']} vehicles -> {stats['counts']}")
            print(f"       Saved annotated video: {output_path}")
            all_results.append(stats)
        except Exception as e:
            print(f"\n  [ERROR] Failed to process {video_name}: {e}")
            all_results.append({
                "video": video_name,
                "status": f"Failed ({type(e).__name__})",
                "total_frames": 0,
                "video_duration_sec": 0.0,
                "processing_time_sec": 0.0,
                "avg_fps": 0.0,
                "total_vehicles": 0,
                "counts": {"Cars": 0, "Motorcycles": 0, "Buses": 0, "Trucks": 0},
                "output_path": "",
            })

    # Save reports
    csv_report_path = os.path.join(results_dir, "traffic_comparison_report.csv")
    json_report_path = os.path.join(results_dir, "traffic_summary.json")

    save_csv_report(all_results, csv_report_path)
    with open(json_report_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)

    total_batch_time = round(time.perf_counter() - t_batch_start, 2)

    # Print summary table to console
    print_comparison_table(all_results)

    print(f"\nBatch processing finished in {total_batch_time}s.")
    print(f"CSV Report:  {os.path.abspath(csv_report_path)}")
    print(f"JSON Report: {os.path.abspath(json_report_path)}")
    print(f"Outputs Dir: {os.path.abspath(output_dir)}")
    print("=" * 65)


if __name__ == "__main__":
    run_batch()
