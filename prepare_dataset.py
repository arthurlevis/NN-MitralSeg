#!/usr/bin/env python3
"""Convert zarr ultrasound data to NN-MitralSeg dataset structure.

Creates: output_path/patient-id_sequence_001/patient-id_sequence_001.avi
Each video contains 4 R-R intervals.
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
import s3fs
from scipy.signal import find_peaks


def open_zarr_group(path: str):
    """Open zarr group from local path or S3."""
    if path.startswith("s3://"):
        profile = os.environ.get("AWS_PROFILE")
        s3 = s3fs.S3FileSystem(profile=profile) if profile else s3fs.S3FileSystem()
        store = s3.get_mapper(path)
    else:
        store = path
    return zarr.open_group(store, mode="r")


def get_obs_group(root):
    """Get observation group from zarr."""
    if "observations" in root:
        return root["observations"]
    elif "ge_ultrasound_rendered_image" in root:
        return root["ge_ultrasound_rendered_image"]
    raise KeyError(f"No observation data found. Available: {list(root.keys())}")


def reconstruct_ecg_signal(ecg_messages: np.ndarray, ecg_timestamps_sec: np.ndarray, ecg_timestamps_nsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct continuous ECG signal from padded message arrays."""
    all_times, all_amplitudes = [], []
    ecg_msg_times = ecg_timestamps_sec + 1e-9 * ecg_timestamps_nsec

    for i, msg in enumerate(ecg_messages):
        n_samples = int(msg[0])
        if n_samples > 0:
            relative_times = msg[1 : n_samples + 1]
            amp_start = n_samples + 1 + 92
            amp_end = 92 + 2 * n_samples + 1
            amplitudes = msg[amp_start:amp_end]
            t0 = ecg_msg_times[i]
            absolute_times = t0 + relative_times - relative_times[0]
            all_times.append(absolute_times)
            all_amplitudes.append(amplitudes)

    times = np.concatenate(all_times)
    amplitudes = np.concatenate(all_amplitudes)
    
    min_len = min(len(times), len(amplitudes))
    times, amplitudes = times[:min_len], amplitudes[:min_len]
    
    valid_mask = ~(np.isnan(amplitudes) | np.isnan(times))
    times, amplitudes = times[valid_mask], amplitudes[valid_mask]
    
    sort_indices = np.argsort(times)
    return times[sort_indices], amplitudes[sort_indices]


def extract_ecg(root) -> tuple[np.ndarray, np.ndarray]:
    """Extract ECG times and amplitudes from zarr."""
    ecg_group = root["ge_ultrasound_ecg_samples"]
    ecg_data = ecg_group["data"][:]
    ts_sec = ecg_group["timestamp_sec"][:]
    ts_nsec = ecg_group["timestamp_nsec"][:]
    return reconstruct_ecg_signal(ecg_data, ts_sec, ts_nsec)


def detect_r_peaks(times: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Detect R-peaks in ECG signal. Returns peak times."""
    sample_rate = 1.0 / np.median(np.diff(times))
    min_distance = int(0.4 * sample_rate)
    peaks, _ = find_peaks(amplitudes, prominence=1.0, distance=min_distance)
    return times[peaks]


def get_frame_timestamps(obs_group, n_frames: int) -> tuple[np.ndarray, float]:
    """Get timestamps for each frame and compute FPS."""
    ts_sec = obs_group["timestamp_sec"][:n_frames]
    ts_nsec = obs_group["timestamp_nsec"][:n_frames]
    times = ts_sec + 1e-9 * ts_nsec
    fps = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 55.0
    return times, fps


def compute_cycle_indices(r_peak_times: np.ndarray, frame_times: np.ndarray) -> list[tuple[int, int]]:
    """Compute frame indices for each R-R interval."""
    cycles = []
    for i in range(len(r_peak_times) - 1):
        start_time, end_time = r_peak_times[i], r_peak_times[i + 1]
        start_idx = np.searchsorted(frame_times, start_time)
        end_idx = np.searchsorted(frame_times, end_time)
        if start_idx < end_idx <= len(frame_times):
            cycles.append((start_idx, end_idx))
    return cycles


def create_sector_mask(height: int, width: int, sector_half_angle_deg: float = 35.0) -> np.ndarray:
    """Create boolean mask for ultrasound sector region."""
    ref_x, ref_y = width // 2, 0
    theta_rad = np.radians(sector_half_angle_deg)
    r_idx = np.arange(height)[:, np.newaxis]
    theta_idx = np.arange(width)[np.newaxis, :]
    angle = -theta_rad + (2 * theta_rad) * (theta_idx / (width - 1))
    x = (ref_x + r_idx * np.sin(angle)).astype(int)
    y = (ref_y + r_idx * np.cos(angle)).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    mask = np.zeros((height, width), dtype=bool)
    mask[y[valid], x[valid]] = True
    return mask


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    """Convert frames to grayscale uint8."""
    if frames.ndim == 4:
        if frames.shape[1] == 1:
            frames = frames[:, 0]
        elif frames.shape[-1] == 1:
            frames = frames[..., 0]
        elif frames.shape[-1] == 3:
            frames = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]).astype(np.uint8)
        else:
            frames = frames[:, 0]
    return frames.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Convert zarr to NN-MitralSeg dataset")
    parser.add_argument("--input-path", required=True, help="Path to zarr (local or s3://...)")
    parser.add_argument("--output-path", required=True, help="Output directory")
    parser.add_argument("--patient-id", required=True, help="Patient ID")
    parser.add_argument("--resize", type=int, nargs=2, default=[112, 112], metavar=("W", "H"), help="Resize to WxH (default: 112 112)")
    parser.add_argument("--num-sequences", type=int, help="Max sequences to extract (default: all)")
    args = parser.parse_args()

    print(f"Opening zarr: {args.input_path}")
    root = open_zarr_group(args.input_path)
    obs_group = get_obs_group(root)
    data = obs_group["data"]
    n_frames = data.shape[0]

    print("Extracting ECG and detecting R-peaks...")
    ecg_times, ecg_amplitudes = extract_ecg(root)
    r_peak_times = detect_r_peaks(ecg_times, ecg_amplitudes)
    print(f"  Found {len(r_peak_times)} R-peaks")

    frame_times, fps = get_frame_timestamps(obs_group, n_frames)
    print(f"  Computed FPS: {fps:.1f}")
    cycles = compute_cycle_indices(r_peak_times, frame_times)
    print(f"  Computed {len(cycles)} R-R intervals")

    # Create sector mask
    h, w = data.shape[1], data.shape[2] if data.ndim >= 3 else data.shape[-1]
    mask = create_sector_mask(h, w)

    # Group cycles into sequences of 4
    intervals_per_seq = 4
    n_sequences = len(cycles) // intervals_per_seq
    if args.num_sequences:
        n_sequences = min(n_sequences, args.num_sequences)
    if n_sequences == 0:
        print(f"Warning: Only {len(cycles)} intervals, need at least {intervals_per_seq}")
        return

    output_base = Path(args.output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    resize_w, resize_h = args.resize

    print(f"Saving {n_sequences} sequences to {output_base}")

    for seq_idx in range(n_sequences):
        start_cycle = seq_idx * intervals_per_seq
        end_cycle = start_cycle + intervals_per_seq
        
        # Get frame range for this sequence
        start_frame = cycles[start_cycle][0]
        end_frame = cycles[end_cycle - 1][1]
        
        # Create sequence name with zero padding
        seq_name = f"{args.patient_id}_sequence_{seq_idx + 1:03d}"
        seq_dir = output_base / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process frames
        frames = to_grayscale(np.array(data[start_frame:end_frame]))
        frames = frames * mask[None, :, :].astype(frames.dtype)
        
        # Write video
        video_path = seq_dir / f"{seq_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (resize_w, resize_h), isColor=False)
        
        for frame in frames:
            frame_resized = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            out.write(frame_resized)
        
        out.release()
        print(f"  {seq_name}: {end_frame - start_frame} frames")

    print(f"Done. Saved {n_sequences} sequences.")


if __name__ == "__main__":
    main()
