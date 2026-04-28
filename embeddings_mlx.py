# SpeakerDiarizationMLX - MLX Speaker Embedding Model
# Defines a lightweight 1D CNN speaker encoder built with Apple MLX.
# Converts waveform segments into 128-dimensional normalized embeddings
# used for speaker clustering and diarization.

import mlx.core as mx
import mlx.nn as nn
import numpy as np

class SimpleSpeakerEncoder(nn.Module):
    """
    Minimal ECAPA-like 1D CNN encoder built with MLX.
    Produces a 128-dimensional normalized speaker embedding vector
    from a raw waveform segment.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(128, 128)

    def __call__(self, x):
        x = mx.relu(self.conv1(x))
        x = mx.relu(self.conv2(x))
        x = mx.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        # Normalize to unit vector for cosine similarity
        x = x / mx.sqrt(mx.sum(x**2, axis=1, keepdims=True) + 1e-8)
        return x


def compute_embeddings(waveform, sr, segments, model, min_duration_sec=0.3):
    """
    Computes speaker embeddings for each detected segment.
    Args:
        waveform: numpy array of the full audio
        sr: sample rate
        segments: list of (start, end) tuples
        model: instance of SimpleSpeakerEncoder
    Returns:
        embeddings: numpy array [n_kept_segments, 128]
        kept_segments: list of (start, end) tuples that produced embeddings
    """
    embs = []
    kept_segments = []
    for start, end in segments:
        s = int(start * sr)
        e = int(end * sr)
        clip = waveform[s:e]
        if len(clip) < sr * min_duration_sec:  # skip very short segments
            continue
        x = mx.array(clip[None, None, :], dtype=mx.float32)
        emb = model(x)
        embs.append(np.array(emb.tolist())[0])
        kept_segments.append((start, end))
    if not embs:
        return np.empty((0, 128), dtype=np.float32), []
    return np.array(embs), kept_segments
