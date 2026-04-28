import numpy as np

from vad import simple_vad


def test_simple_vad_detects_two_speech_regions():
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    speech = np.ones(sr, dtype=np.float32) * 0.2
    waveform = np.concatenate([silence, speech, silence, speech])

    segments = simple_vad(waveform, sr, frame_ms=100, threshold=0.01)

    assert len(segments) == 2
    assert 1.0 <= segments[0][0] <= 1.1
    assert 1.9 <= segments[0][1] <= 2.1
    assert 3.0 <= segments[1][0] <= 3.1
    assert 3.9 <= segments[1][1] <= 4.1


def test_vad_merges_close_gaps():
    sr = 16000
    speech = np.ones(int(sr * 0.4), dtype=np.float32) * 0.2
    short_pause = np.zeros(int(sr * 0.1), dtype=np.float32)
    waveform = np.concatenate([speech, short_pause, speech])

    segments = simple_vad(
        waveform,
        sr,
        frame_ms=50,
        threshold=0.01,
        merge_gap_ms=150,
        min_speech_ms=100,
    )

    assert len(segments) == 1
    assert 0.0 <= segments[0][0] <= 0.05
    assert 0.85 <= segments[0][1] <= 0.95


def test_vad_filters_short_segments_by_duration():
    sr = 16000
    short_speech = np.ones(int(sr * 0.1), dtype=np.float32) * 0.2
    silence = np.zeros(int(sr * 0.2), dtype=np.float32)
    long_speech = np.ones(int(sr * 0.5), dtype=np.float32) * 0.2
    waveform = np.concatenate([short_speech, silence, long_speech])

    segments = simple_vad(
        waveform,
        sr,
        frame_ms=50,
        threshold=0.01,
        merge_gap_ms=0,
        min_speech_ms=250,
    )

    assert len(segments) == 1
    assert 0.25 <= segments[0][0] <= 0.35
    assert 0.75 <= segments[0][1] <= 0.85


