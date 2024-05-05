import numpy as np
from collections import defaultdict

metric_map = {
    'log_probs': 'Log Probability',
    'neg_log_probs': 'Neg Log Probability',
    'cosine_similarity': 'Cosine Similarity',
    'dpp_score': 'DPP Score',
    'lp_distance': 'LP Distance',
    'temperature': 'Temperature',
}


def cosine_similarity_score(embeddings):
    # Compute cosine similarity between all pairs of embeddings
    # Return the average of the pairwise cosine similarities
    n, d = embeddings.shape
    S = (embeddings @ embeddings.T) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1).T)
    score = np.tril(S, -1)
    return score.sum() / (n * (n - 1) / 2)

def lp_distance_score(embeddings):
    # Compute Lp distance between all pairs of embeddings
    # Return the average of the pairwise Lp distances
    n, d = embeddings.shape
    S = np.linalg.norm(embeddings[:, None] - embeddings[None, :], ord=2, axis=2)
    score = np.tril(S, -1)
    return score.sum() / (n * (n - 1) / 2)

def dpp_score(embeddings):
    # Compute the Determinantal Point Process (DPP) score
    # Return the DPP score
    n, d = embeddings.shape
    K = embeddings @ embeddings.T / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1).T)
    L = np.linalg.cholesky(K + 1e-5 * np.eye(n))
    score = np.linalg.det(L @ L.T)
    return score

def compute_metrics(output, embedding_model):
    ret = {}
    ret['log_probs'] = output['logprobs'].sum().item()
    ret['neg_log_probs'] = -ret['log_probs']

    embeddings = embedding_model.encode(output['generated_text'])
    ret['cosine_similarity'] = cosine_similarity_score(embeddings)
    ret['dpp_score'] = dpp_score(embeddings)
    ret['lp_distance'] = lp_distance_score(embeddings)
    return ret


class ResultCollector:
    """
        Collects metrics for a given set of related predictions.
    """
    def __init__(self):
        self.perf_metrics = defaultdict(list)

    def reset(self):
        self.perf_metrics = defaultdict(list)

    def add(self, perf_metrics: dict):
        for key, value in perf_metrics.items():
            self.perf_metrics[key].append(value)

    def get_average(self, metric, std=False):
        if metric not in self.perf_metrics.keys():
            raise ValueError(f"no metric: {metric}")

        score = np.array(self.perf_metrics[metric])
        if std:
            return score.mean().item(), score.std().item()
        else:
            return score.mean().item()
    
    def get_df(self):
        import pandas as pd
        return pd.DataFrame(self.perf_metrics)

    def get_summarized_results(self, std=False):
        ret = {}
        for metric in self.perf_metrics.keys():
            ret[metric] = self.get_average(metric)
        return ret