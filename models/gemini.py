import os
import google.generativeai as gg
from .base import BaseModel

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY != "":
    gg.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY is not set")

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


class Gemini(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = gg.GenerativeModel(self.backend)

    def __call__(self, prompt, num_samples=1, stop_sequences=None, temperature=None, top_p=None, max_tokens=None):
        """
            prompt: string
        """
        output = []
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        max_tokens = max_tokens or self.max_tokens

        while num_samples > 0:
            cnt = min(num_samples, 1)
            num_samples -= cnt
            generation_config = gg.GenerationConfig(
                max_output_tokens=max_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                temperature=temperature,
                candidate_count=cnt,
            )
            response = self.model.generate_content(prompt,
                                                   generation_config=generation_config,
                                                   safety_settings=safety_settings)
            output.extend([rcp.text for r in response.candidates for rcp in r.content.parts])
            
        self.add_to_history(prompt, output)
        return output