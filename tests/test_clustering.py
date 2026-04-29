import numpy as np

from clustering import cluster_embeddings


def test_cluster_embeddings_handles_empty_input():
    embeddings = np.empty((0, 128), dtype=np.float32)
    labels = cluster_embeddings(embeddings, n_clusters=2)
    assert labels.size == 0


def test_cluster_embeddings_caps_cluster_count_to_sample_count():
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = cluster_embeddings(embeddings, n_clusters=4)
    assert labels.shape[0] == 2
    assert set(labels.tolist()) == {0, 1}

