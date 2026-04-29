# SpeakerDiarizationMLX
# Main entry point of the project.
# Loads an audio file, performs voice activity detection (VAD),
# extracts speaker embeddings with MLX, clusters speakers,
# and visualizes who spoke when.

import argparse
from audio_utils import load_audio
from vad import simple_vad
from clustering import cluster_embeddings
from visualization import plot_diarization
from embeddings_mlx import SimpleSpeakerEncoder, compute_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Speaker diarization with MLX")
    parser.add_argument(
        "--audio-path",
        default="sample_audio/conversation.wav",
        help="Input audio file (.wav or .m4a)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--vad-frame-ms", type=int, default=30)
    parser.add_argument("--vad-threshold", type=float, default=0.02)
    parser.add_argument("--vad-merge-gap-ms", type=int, default=200)
    parser.add_argument("--vad-min-speech-ms", type=int, default=300)
    parser.add_argument("--min-segment-sec", type=float, default=0.3)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    audio, sr = load_audio(args.audio_path, target_sr=args.sample_rate)
    print(f"Loaded {len(audio)} samples at {sr} Hz")

    segments = simple_vad(
        audio,
        sr,
        frame_ms=args.vad_frame_ms,
        threshold=args.vad_threshold,
        merge_gap_ms=args.vad_merge_gap_ms,
        min_speech_ms=args.vad_min_speech_ms,
    )
    print(f"Detected {len(segments)} segments")
    if not segments:
        print("No speech segments detected. Try lowering --vad-threshold.")
        return

    model = SimpleSpeakerEncoder()
    embeddings, kept_segments = compute_embeddings(
        audio,
        sr,
        segments,
        model,
        min_duration_sec=args.min_segment_sec,
    )
    print(f"Generated {embeddings.shape[0]} embeddings")
    if embeddings.shape[0] == 0:
        print("No embeddings generated. Try lowering --min-segment-sec.")
        return

    labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters)

    for (start, end), label in zip(kept_segments, labels):
        print(f"Speaker {label}: {start:.1f}-{end:.1f}s")

    if not args.no_plot:
        plot_diarization(kept_segments, labels)


if __name__ == "__main__":
    main()
