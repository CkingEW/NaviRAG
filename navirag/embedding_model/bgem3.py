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


class BGEEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        self.embedding_model_name = embedding_model_name or "BAAI/bge-m3"
        logger.debug(f"Initializing {self.__class__.__name__} with model: {self.embedding_model_name}")

        self._init_embedding_config()

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True, device_map="auto")
        self.embedding_dim = self.model.config.hidden_size
        self.model.eval()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": self.global_config.embedding_model_dtype,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)
        max_length = params.pop("max_length", 8192)
        normalize = self.embedding_config.norm       

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        results = []
        pbar = tqdm(total=len(texts), desc="Batch Encoding")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                results.append(embeddings.cpu())

            pbar.update(len(batch_texts))

        pbar.close()
        return torch.cat(results, dim=0).numpy()
