from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]

        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class QwenEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        self.embedding_model_name = embedding_model_name or "Qwen/Qwen3-Embedding-4B"
        logger.debug(f"Initializing {self.__class__.__name__} with model: {self.embedding_model_name}")

        self._init_embedding_config()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, 
            trust_remote_code=True, 
            padding_side='left'
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModel.from_pretrained(
            self.embedding_model_name, 
            trust_remote_code=True, 
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32 
        )
        
        self.embedding_dim = self.model.config.hidden_size
        self.model.eval()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {}, 
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "", 
                "batch_size": self.global_config.embedding_batch_size,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def batch_encode(self, texts: List[str], instruction: str = None, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)
        max_length = params.pop("max_length", 8192) 
        normalize = self.embedding_config.norm

        if instruction:
            new_texts = []
            for text in texts:
                new_text = f"Instruct: {instruction}\nQuery: {text}"
                new_texts.append(new_text)
            texts = new_texts

        results = []
        pbar = tqdm(total=len(texts), desc="Batch Encoding (Qwen)")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                results.append(embeddings.cpu())

            pbar.update(len(batch_texts))

        pbar.close()
        return torch.cat(results, dim=0).numpy()