# SpeakerDiarizationMLX

A minimal, educational example of **speaker diarization** built with **Apple MLX**.
It detects who spoke when in an audio recording by combining **Voice Activity Detection (VAD)**,
a small **MLX-based speaker embedding model**, and **unsupervised clustering**.

---

## 🚀 Features

- 🔊 Load and process `.wav` or `.m4a` conversation recordings
- 🎤 Detect voice activity using an energy-based VAD
- ✂️ Post-process VAD with configurable gap-merge and minimum speech duration
- 🧬 Extract 128-dimensional speaker embeddings with a small 1D CNN in MLX
- 👥 Cluster speakers using KMeans to identify who spoke when
- 📊 Visualize the result in a clean matplotlib timeline plot

---

## 🧩 Project structure

```text
main.py                             # Entry point / CLI
audio_utils.py                      # Audio loading and resampling
vad.py                              # Voice Activity Detection
embeddings_mlx.py                   # MLX speaker encoder
clustering.py                       # Clustering of embeddings
visualization.py                    # Plot who spoke when
generate_multilingual_conversations.py
pyproject.toml                      # Project metadata + pytest config
Makefile                            # Common dev/run shortcuts
.github/workflows/test.yml          # CI test workflow
tests/                              # Basic unit tests
sample_audio/                       # Directory for test files
```

## ⚙️ Installation

### Option 1 - Using **uv** (recommended)

```bash
git clone https://github.com/marcgeld/SpeakerDiarizationMLX.git
cd SpeakerDiarizationMLX
uv sync --extra dev
```

Run commands through uv:

```bash
uv run diarize --help
uv run pytest -q
```

### Makefile shortcuts (uv workflow)

```bash
make help
make sync
make run
make run ARGS="--audio-path sample_audio/conversation_english.wav --no-plot"
make test
make clean
```

### Option 2 - Using **pip**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
```

## ▶️ Usage

1. Place a `.wav` or `.m4a` file inside `sample_audio/`, or generate test files:

```bash
python generate_multilingual_conversations.py
```

> If you use `generate_multilingual_conversations.py`, you may need
> [ffmpeg](https://www.ffmpeg.org/) on macOS:
> `brew install ffmpeg`

2. Run diarization on the default sample path:

```bash
uv run diarize
```

3. Or run with explicit options:

```bash
uv run diarize --audio-path sample_audio/conversation_english.wav --n-clusters 2 --vad-threshold 0.02
```

Useful VAD post-processing flags:

```bash
uv run diarize --vad-merge-gap-ms 200 --vad-min-speech-ms 300
```

4. Run tests:

```bash
uv run pytest -q
```

Example output:

```text
Loaded 240000 samples at 16000 Hz
Detected 12 segments
Generated 12 embeddings
Speaker 0: 0.5-1.8s
Speaker 1: 2.0-3.4s
```