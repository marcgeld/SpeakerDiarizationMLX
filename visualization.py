# SpeakerDiarizationMLX - Visualization
# Uses matplotlib to display a simple speaker timeline plot.
# Each speaker is represented by a color-coded horizontal line
# showing when that person was speaking.

import matplotlib.pyplot as plt

def plot_diarization(segments, labels):
    """
    Plot a simple speaker diarization timeline.
    Each speaker is assigned a color-coded horizontal line
    indicating when they were speaking.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    for (start, end), label in zip(segments, labels):
        ax.plot([start, end], [label, label], linewidth=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speaker ID")
    ax.set_title("Speaker Diarization (MLX Prototype)")
    plt.tight_layout()
    plt.show()