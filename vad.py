# SpeakerDiarizationMLX - Voice Activity Detection (VAD)
# Implements a simple energy-based VAD algorithm.
# Splits the audio into short frames and detects segments
# where speech energy exceeds a threshold.

import numpy as np

def _merge_close_segments(segments, merge_gap_sec):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def simple_vad(
    waveform,
    sr,
    frame_ms=30,
    threshold=0.02,
    merge_gap_ms=200,
    min_speech_ms=300,
):
    """
    Simple energy-based Voice Activity Detection (VAD).
    Splits the audio into frames and marks speech segments
    when average frame energy exceeds the threshold.
    Post-processes segments by merging close gaps and removing
    short speech segments.
    Returns:
        List of (start, end) tuples in seconds.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    energies = [
        np.mean(np.square(waveform[i:i+frame_len]))
        for i in range(0, len(waveform), frame_len)
    ]
    speech_flags = np.array(energies) > threshold

    segments = []
    in_seg = False
    start = 0
    for i, flag in enumerate(speech_flags):
        if flag and not in_seg:
            in_seg = True
            start = i * frame_ms / 1000
        elif not flag and in_seg:
            in_seg = False
            end = i * frame_ms / 1000
            segments.append((start, end))
    if in_seg:
        segments.append((start, len(waveform)/sr))

    merge_gap_sec = max(0.0, merge_gap_ms / 1000)
    min_speech_sec = max(0.0, min_speech_ms / 1000)
    segments = _merge_close_segments(segments, merge_gap_sec)
    if min_speech_sec > 0:
        segments = [
            (start, end) for start, end in segments
            if end - start >= min_speech_sec
        ]
    return segments
