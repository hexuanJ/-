"""
k-Reciprocal Re-Ranking (CVPR 2017)
Reference: https://arxiv.org/abs/1701.08398

Usage:
    from reranking import re_ranking
    reranked_dist = re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3)
    # reranked_dist[i, j] is the re-ranked distance between query i and gallery j
"""
import numpy as np

# Minimum overlap fraction required to include a neighbour's reciprocal set
_OVERLAP_THRESHOLD = 2.0 / 3


def _pairwise_euclidean_distance(x, y):
    """Compute pairwise squared Euclidean distance between rows of x and y."""
    xx = (x ** 2).sum(axis=1, keepdims=True)
    yy = (y ** 2).sum(axis=1, keepdims=True)
    dist = xx + yy.T - 2 * x @ y.T
    dist = np.maximum(dist, 0.0)
    return dist


def re_ranking(query_feat, gallery_feat, k1=20, k2=6, lambda_value=0.3):
    """k-Reciprocal Re-Ranking.

    Args:
        query_feat:   numpy array (Nq, D) of L2-normalised query features.
        gallery_feat: numpy array (Ng, D) of L2-normalised gallery features.
        k1:           k for initial k-reciprocal neighbours.
        k2:           k for query-expansion.
        lambda_value: weight of the original distance term.

    Returns:
        dist: numpy array (Nq, Ng) of re-ranked distances (lower = more similar).
    """
    query_feat = np.array(query_feat, dtype=np.float32)
    gallery_feat = np.array(gallery_feat, dtype=np.float32)

    nq = query_feat.shape[0]
    ng = gallery_feat.shape[0]
    all_feat = np.concatenate([query_feat, gallery_feat], axis=0)  # (nq+ng, D)
    n_all = nq + ng

    # Original pairwise distance (query vs gallery)
    original_dist = _pairwise_euclidean_distance(query_feat, gallery_feat)
    # Normalise to [0, 1]
    original_dist = original_dist / (original_dist.max() + 1e-12)

    # Full pairwise distance for k-reciprocal computation
    all_dist = _pairwise_euclidean_distance(all_feat, all_feat)

    # Sort indices by ascending distance
    sorted_idx = np.argsort(all_dist, axis=1)

    def k_reciprocal_neigh(i, k):
        """Return indices within k-reciprocal neighbours of i."""
        forward_k = sorted_idx[i, 1:k + 1]  # exclude self
        reciprocal = []
        for j in forward_k:
            back_k = sorted_idx[j, 1:k + 1]
            if i in back_k:
                reciprocal.append(j)
        return np.array(reciprocal, dtype=np.int32)

    # Build Jaccard-distance matrix
    V = np.zeros((n_all, n_all), dtype=np.float32)

    for i in range(n_all):
        R = k_reciprocal_neigh(i, k1)
        if len(R) == 0:
            continue

        # Query expansion: add neighbours of neighbours
        R_exp = list(R)
        for j in R:
            R2 = k_reciprocal_neigh(j, max(int(np.round(k1 / 2)), 1))
            if len(R2) > 0:
                # Include if enough overlap
                inter = len(set(R.tolist()) & set(R2.tolist()))
                if inter >= _OVERLAP_THRESHOLD * len(R2):
                    R_exp.extend(R2.tolist())
        R_exp = np.unique(R_exp).astype(np.int32)

        # Weight by similarity (Gaussian kernel)
        weight = np.exp(-all_dist[i, R_exp])
        V[i, R_exp] = weight / (weight.sum() + 1e-12)

    # Query expansion with k2
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n_all):
            neighbors = sorted_idx[i, :k2]
            V_qe[i] = V[neighbors].mean(axis=0)
        V = V_qe

    # Jaccard distance: only query-gallery portion
    V_q = V[:nq]    # (nq, n_all)
    V_g = V[nq:]    # (ng, n_all)

    # jaccard_dist[i,j] = 1 - |V_q[i] ∩ V_g[j]| / |V_q[i] ∪ V_g[j]|
    # Computed efficiently via minimum
    min_sum = np.zeros((nq, ng), dtype=np.float32)
    for i in range(nq):
        min_sum[i] = np.minimum(V_q[i:i+1], V_g).sum(axis=1)

    union_sum = V_q.sum(axis=1, keepdims=True) + V_g.sum(axis=1) - min_sum
    jaccard_dist = 1.0 - min_sum / (union_sum + 1e-12)
    jaccard_dist = np.maximum(jaccard_dist, 0.0)

    # Final distance
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    return final_dist
