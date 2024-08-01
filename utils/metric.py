import numpy as np


def compute_corrcoef_std(A, B):
    assert A.shape == B.shape, f"Input shape not match, A:{A.shape}, B:{B.shape}"
    A = A.reshape(-1)
    B = B.reshape(-1)
    if (A == B).all():
        return 1.0, 0.0
    coef = np.corrcoef(A, B)[0][1]
    std_dev = np.std(A - B)

    return coef, std_dev


def cosine_similarity(A, B):
    assert A.shape == B.shape, f"Input shape not match, A:{A.shape}, B:{B.shape}"
    A = A.reshape(-1)
    B = B.reshape(-1)
    if (A == B).all():
        return 1.0
    cosine_sim = np.dot(A, B) / np.maximum(np.linalg.norm(A) * np.linalg.norm(B), 1e-30)
    return cosine_sim


def normalized_relative_error(A, B):
    """
    A as groud truth.
    """
    assert A.shape == B.shape, f"Input shape not match, A:{A.shape}, B:{B.shape}"
    A = A.reshape(-1)
    B = B.reshape(-1)
    norm_rel_err = np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-30)
    return norm_rel_err


def diff_max_mean(A, B):
    assert A.shape == B.shape, f"Input shape not match, A:{A.shape}, B:{B.shape}"
    diff = np.abs(A - B)
    return diff.max(), diff.mean()
