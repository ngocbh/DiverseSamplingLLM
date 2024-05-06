import os
import os.path as osp
import pandas as pd
import numpy as np
import fire
import pickle
import time
import torch
import random

from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial
from sentence_transformers import SentenceTransformer

from decoding import naive_decoding, dpp_decoding
from models.mistral import Mistral
from metrics import compute_metrics, ResultCollector
from visualize import plot


def preprocess_rocstories(df, max_samples=1000):
    STORY_GEN_TEMPLATE = "<s>[INST] Complete the last two sentences of the following story.[/INST] Title: {}\nStory: {}\nLast two sentences:"
    sample_df = df.sample(max_samples)
    data = []
    for index, row in sample_df.iterrows():
        title = row['storytitle']
        story = row['sentence1'] + ' ' + row['sentence2'] + ' ' + row['sentence3']
        prompt = STORY_GEN_TEMPLATE.format(title, story)
        data.append({
            'prompt': prompt,
            'completion': row['sentence4'] + ' ' + row['sentence5']
        })
    return data


def load_rocstories(max_samples=1000):
    df = pd.read_csv('datasets/ROCStories/dataset_2017.csv')
    dataset = preprocess_rocstories(df, max_samples=max_samples)
    return dataset


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(dataset, decoding_fn, embedding_model, k, num_seeds=1):
    rescol = ResultCollector()
    for sample in tqdm(dataset):
        sample_rescol = ResultCollector()
        for i in range(num_seeds):
            start_time = time.time()
            out = decoding_fn(sample['prompt'].strip(), k = k)
            end_time = time.time()
            metrics = compute_metrics(out, embedding_model)
            metrics['time'] = end_time - start_time
            sample_rescol.add(metrics)
        rescol.add(sample_rescol.get_summarized_results())
    return rescol

def evaluate_varying_param(dataset, decoding_fn, language_model, embedding_model, k, param_to_vary, num_seeds=1, **kwargs):
    param_name = param_to_vary['name']
    min_value = param_to_vary['min']
    max_value = param_to_vary['max']
    step = param_to_vary['step']
    param_values = np.arange(min_value, max_value, step)

    rescol = ResultCollector()
    for value in param_values:
        param_dict = {param_name: value}
        param_dict.update(kwargs)
        partial_decoding_fn = partial(decoding_fn, language_model=language_model, embedding_model=embedding_model, **param_dict)
        res = evaluate(dataset, partial_decoding_fn, embedding_model, k, num_seeds)
        res_dict = res.get_summarized_results()
        res_dict[param_name] = value
        rescol.add(res_dict)
    return rescol

def naive_decoding_fn(wdir, dataset, language_model, embedding_model, args):
    if not os.path.exists(osp.join(wdir, 'naive_res.pkl')) or args.rerun:
        # running the naive decoding
        print("Running Naive Decoding...")
        param_to_vary = {
            'name': 'temperature',
            'min': 0.001,
            'max': 1.1,
            'step': 0.2
        }
        naive_res = evaluate_varying_param(dataset, naive_decoding, language_model, embedding_model, args.k, param_to_vary=param_to_vary, num_seeds=args.num_seeds)
        pickle.dump(naive_res, open(osp.join(wdir, 'naive_res.pkl'), 'wb'))
    else:
        naive_res = pickle.load(open(osp.join(wdir, 'naive_res.pkl'), 'rb'))
    return naive_res


def dpp_decoding_fn(wdir, dataset, language_model, embedding_model, args, method='sampling'):
    if not os.path.exists(osp.join(wdir, f'{method}_res.pkl')) or args.rerun:
        # running the naive decoding
        print("Running DPP Decoding...")
        param_to_vary = {
            'name': 'gamma',
            'min': 0.001,
            'max': 1.1,
            'step': 0.2
        }
        dpp_params = {
            'n': args.dpp_params_n,
        }
        partial_dpp_decoding = partial(dpp_decoding, method=method)
        dpp_res = evaluate_varying_param(dataset, partial_dpp_decoding, language_model, embedding_model,
                                         k=args.k, param_to_vary=param_to_vary, num_seeds=args.num_seeds, **dpp_params)
        pickle.dump(dpp_res, open(osp.join(wdir, f'{method}_res.pkl'), 'wb'))
    else:
        dpp_res = pickle.load(open(osp.join(wdir, f'{method}_res.pkl'), 'rb'))
    return dpp_res


method_map = {
    'naive': naive_decoding_fn,
    'dpp_sampling': dpp_decoding_fn,
    'dpp_map_ls': partial(dpp_decoding_fn, method='map_ls'),
}

method_name_map = {
    'naive': 'Temperature Sampling',
    'dpp_sampling': 'DPP Sampling',
    'dpp_map_ls': 'DPP MAP Inference',
}


@dataclass
class Args:
    max_samples: int = 50
    k: int = 5
    num_seeds: int = 5
    run_id: int = 0
    methods: tuple[str] = ('naive', 'dpp_sampling', 'dpp_map_ls')
    rerun: bool = False
    seed: int = 42

    # DPP specific params
    dpp_params_n: int = 50

    def update(self, **kwargs):
        methods = kwargs.get('methods', self.methods)
        if isinstance(methods, str):
            methods = (methods,)
        kwargs['methods'] = methods
        for key, value in kwargs.items():
            setattr(self, key, value)


def main(**kwargs):
    args = Args()
    args.update(**kwargs)
    if args.run_id == 0:
        args.max_samples = 3
        args.num_seeds = 1
        args.rerun = True
        args.k = 2
        args.dpp_params_n = 10
    wdir = f'./results/{args.run_id}/'

    print(f"Running with args: {args}")
    seed_everything(args.seed)

    os.makedirs(wdir, exist_ok=True)
    language_model = Mistral()
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    dataset = load_rocstories(max_samples=args.max_samples)

    res = []
    for method in args.methods:
        method_fn = method_map[method]
        res.append(method_fn(wdir, dataset, language_model, embedding_model, args))

    method_names = [method_name_map[method] for method in args.methods]

    plot(res, method_names, 'cosine_similarity', 'neg_log_probs', y_scale='log', save_path=osp.join(wdir, 'cosine_log_probs.png'))
    plot(res, method_names, 'dpp_score', 'neg_log_probs', y_scale='log', save_path=osp.join(wdir, 'dpp_log_probs.png'))
    plot(res, method_names, 'lp_distance', 'neg_log_probs', y_scale='log', save_path=osp.join(wdir, 'lp_dist_log_probs.png'))


if __name__ == '__main__':
    fire.Fire(main)
