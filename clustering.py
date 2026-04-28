# SpeakerDiarizationMLX - Speaker Clustering
# Groups speaker embeddings using KMeans or other clustering algorithms.
# Assigns a speaker label to each detected speech segment.

import numpy as np
from sklearn.cluster import KMeans

def cluster_embeddings(embeddings, n_clusters=2):
    """
    Cluster speaker embeddings using KMeans.
    Args:
        embeddings: numpy array [n_segments, embedding_dim]
    Returns:
        cluster labels: numpy array [n_segments]
    """
    if embeddings.size == 0:
        return np.array([], dtype=int)
    effective_clusters = min(n_clusters, embeddings.shape[0])
    model = KMeans(n_clusters=effective_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels