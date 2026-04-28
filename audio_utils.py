# SpeakerDiarizationMLX - Audio utilities
# Handles audio loading, resampling, and mono conversion.
# Provides functions to return waveform data as NumPy arrays
# ready for processing by the MLX model.

import torchaudio
import numpy as np

def load_audio(path, target_sr=16000):
    """
    Loads an audio file and converts it to mono with a fixed sample rate.
    Returns:
        waveform: numpy array [samples]
        sr: sampling rate
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono
    return waveform.mean(dim=0).numpy(), target_sr
