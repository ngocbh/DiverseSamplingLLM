import numpy as np
import torch

from .base import BaseModel


class Mistral(BaseModel):
    def __init__(self, backend="mistralai/Mistral-7B-Instruct-v0.2", **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(backend, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(backend)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.device
        super().__init__(backend, **kwargs)
    
    def __call__(self, prompt, **kwargs):
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 200)
        stop_sequences = kwargs.get('stop_sequences', None)
        num_samples = kwargs.get('num_samples', 1)
        transition_scores = kwargs.get('logprobs', True)
        output_scores = kwargs.get('output_scores', True)
        do_sample = kwargs.get('do_sample', True)
        
        input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **input,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=max_tokens,
            return_dict_in_generate=True,
            use_cache=True,
            output_scores=output_scores,
        )

        transition_scores = self.model.compute_transition_scores(
            output.sequences,
            output.scores,
            normalize_logits=True
        )

        input_length = 1 if self.model.config.is_encoder_decoder else input.input_ids.shape[1]
        generated_tokens = output.sequences[:, input_length:]
        output_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # for batch generation, the output of some sequences may include special padding tokens, which is also included in the transition scores
        # Those special tokens usually get a score of -inf, making the sum of the transition scores -inf
        # A temporary fix is to replace -inf with 0.0, but it's not safe
        log_probs = torch.sum(torch.nan_to_num(transition_scores, neginf=0.0), dim=1).cpu().numpy()

        return {
            "generated_text": output_str,
            "logprobs": log_probs,
        }
      