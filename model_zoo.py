# Contains code to load a model
import enum
import subprocess

import torch
from transformers import pipeline, AutoTokenizer
from FlagEmbedding import FlagModel

from config import Config


class Models(enum.Enum):
    LLAMA_70B_CHAT = 'meta-llama/Llama-2-70b-chat-hf'


class ModelContainer:
    def __init__(self, model: Models, is_v2_llama):
        self.is_v2_llama = is_v2_llama
        self.tokenizer = AutoTokenizer.from_pretrained(model.value)

        self.pipeline = pipeline(
            'text-generation',
            model=model.value,  # model.value is MUCH faster (as above)
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        subprocess.run(['nvidia-smi'])

    def generate(self, prompt):
        # Better to encapsulate this here so that the
        # calling function doesn't need to be aware of the tokenizer
        max_length = 4096 if self.is_v2_llama else 2048

        response = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_length,
            prefix=Config.system_prompt
        )
        return response


class EmbeddingModel:
    @classmethod
    def load_model(cls, identifier=None):

        model = FlagModel(
            'BAAI/bge-large-en-v1.5',
            use_fp16=True,
        )

        return model
