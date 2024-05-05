from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self,
                 backend,
                 temperature=0.1,
                 top_p=0.9,
                 max_tokens=1000):
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.history = []
        self.num_queries = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self.history = []
        self.num_queries = 0

    def add_to_history(self, prompt, response):
        self.history.append({
            "prompt": prompt,
            "response": response,
        })
        self.num_queries += 1

    @abstractmethod
    def __call__(self):
        raise NotImplementedError