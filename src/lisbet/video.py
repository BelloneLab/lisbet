import random
import subprocess
from collections import defaultdict
from datetime import timedelta
from itertools import groupby
from pathlib import Path
from time import gmtime, strftime

import numpy as np
from tqdm.auto import tqdm, trange


def frame2time(frame_count, fps=30):
    td = timedelta(seconds=(frame_count / fps))
    return strftime("%H:%M:%S", gmtime(td.seconds)) + f".{int(td.microseconds/1000)}"


def split_motifs(
    seq_keys,
    predictions,
    lengths,
    video_path,
    fps,
    video_keydrop=None,
    video_sample=None,
    output_path=None,
    seed=None,
    scale=None,
    focus_mode=False,
    min_duration=None,
    max_duration=None,
):
    """ """
    random.seed(seed)

    # Find number of states
    num_states = int(np.max(predictions)) + 1

    video_chunks = defaultdict(list)

    for seq_idx in trange(len(seq_keys), desc="Processing videos"):
        key = seq_keys[seq_idx]

        # Find video path
        filename = f"{Path(key).relative_to(video_keydrop)}.mp4"
        orig_video_path = sorted(Path(video_path).rglob(filename))
        assert len(orig_video_path) == 1
        orig_video_path = orig_video_path[0]

        # Find data indices
        seq_start = sum(lengths[:seq_idx])
        seq_stop = seq_start + lengths[seq_idx]

        # Extract video predictions
        pred = predictions[seq_start:seq_stop]

        # Make list of MotifID: [(bout_start, bout_stop), ...]
        bouts_indices = defaultdict(list)
        events = [(k, sum(1 for i in g)) for k, g in groupby(pred)]
        bout_start = 0
        for k, s in events:
            # Validate bout duration, if requested
            if (min_duration is None or s / fps > min_duration) and (
                max_duration is None or s / fps < max_duration
            ):
                bouts_indices[k].append((bout_start, bout_start + s))
            bout_start += s

        if video_sample is not None:
            for motif_id in range(num_states):
                sample_size = int(len(bouts_indices[motif_id]) * video_sample)
                bouts_indices[motif_id] = random.sample(
                    bouts_indices[motif_id], sample_size
                )

        # Export video bouts
        for motif_id, bouts in tqdm(
            bouts_indices.items(), leave=False, desc="Processing bouts"
        ):
            for bout_idx, (bout_start, bout_stop) in enumerate(bouts):
                # Create output directory
                bout_video_path = (
                    Path(output_path) / f"{key}/Motif{motif_id}/Bout{bout_idx}.mp4"
                )
                bout_video_path.parent.mkdir(parents=True, exist_ok=True)

                # Store video path for later (merging chunks)
                video_chunks[motif_id].append(str(bout_video_path))

                # Select t_start and t_stop
                if focus_mode:
                    # Window of 6 seconds, centered to the beginning of the bout
                    t_start = frame2time(max(bout_start - 3 * fps, 0))
                    t_stop = frame2time(bout_start + 3 * fps)
                else:
                    # Window of 1 second before and after the bout
                    t_start = frame2time(max(bout_start - fps, 0))
                    t_stop = frame2time(bout_stop + fps)

                # Base command
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "warning",
                    "-i",
                    orig_video_path,
                    "-ss",
                    t_start,
                    "-to",
                    t_stop,
                ]

                # Default
                if scale is None:
                    cmd.extend(
                        [
                            "-c",
                            "copy",
                            bout_video_path,
                        ]
                    )

                # Scale video to desired resolution (slow)
                else:
                    cmd.extend(
                        [
                            "-vf",
                            f"scale={scale},setsar=1",
                            "-c:v",
                            "libx264",  # Re-encode video using libx264
                            "-c:a",
                            "copy",  # Copy the audio stream as is
                            bout_video_path,
                        ]
                    )

                # Trim bout video
                subprocess.run(cmd)

    return video_chunks


def motif_collage(video_chunks, output_path=None, seed=None):
    """ """
    random.seed(seed)

    for motif_id in tqdm(video_chunks):
        # Shuffle list (workaround to avoid in place shuffling)
        chunks_paths = random.sample(
            video_chunks[motif_id], len(video_chunks[motif_id])
        )

        output_fname = Path(output_path) / f"examples_motif{motif_id}.mp4"

        # Run ffmpeg
        cmd = (
            ["ffmpeg", "-y", "-loglevel", "warning"]
            + [
                val
                for tup in zip(["-i"] * len(chunks_paths), chunks_paths)
                for val in tup
            ]
            + [
                "-filter_complex",
                "".join([f"[{i}:v:0]" for i in range(len(chunks_paths))])
                + f"concat=n={len(chunks_paths)}:v=1[outv]",
                "-map",
                "[outv]",
                output_fname,
            ]
        )
        subprocess.run(cmd)
