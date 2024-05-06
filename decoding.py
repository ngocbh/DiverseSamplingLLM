import numpy as np
from dpp import map_inference_dpp_greedy, map_inference_dpp_local_search_2, dpp_sampling


def naive_decoding(prompt, language_model, embedding_model, k, temperature=0.7):
    output = language_model(
        prompt,
        temperature=temperature,
        num_samples=k,
    )

    return output


def dpp_decoding(prompt, language_model, embedding_model, k, method='sampling', n=100, kernel_width=50, gamma=0.1, temperature=0.5):
    output = language_model(
        prompt,
        temperature=temperature,
        num_samples=n,
    )
    embeddings = embedding_model.encode(output['generated_text'])
    log_probs = output['logprobs']
    D = np.diag(np.exp(- log_probs ** 2 / kernel_width ** 2))
    S = (embeddings @ embeddings.T) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1).T)
    gamma = 0.2
    L = gamma * S + (1 - gamma) * D

    if method == 'sampling':
        # there is a bug in the dpp sampling code, potentially due to the matrix being singular
        try:
            dpp_samples = dpp_sampling(L, k)
        except:
            dpp_samples = np.random.choice(len(embeddings), k, replace=False)
    elif method == 'map_ls':
        dpp_samples, _, _, _, _, _ = map_inference_dpp_local_search_2(L, k)
    elif method == 'map_greedy':
        dpp_samples, _ = map_inference_dpp_greedy(L, k)

    generated_texts = [output['generated_text'][i] for i in dpp_samples]
    logprogs = output['logprobs'][dpp_samples]

    return {
        'generated_text': generated_texts,
        'logprobs': logprogs,
    }
