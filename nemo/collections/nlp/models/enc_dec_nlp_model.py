from abc import abstractmethod
from dataclasses import MISSING, dataclass, asdict
import math
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from typing import Dict, Optional, Union

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.nlp_model import NLPModel


@dataclass
class TokenizerConfig:
    tokenizer_name: str = MISSING
    tokenizer_model: Optional[str] = None
    vocab_file: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None


@dataclass
class EmbeddingConfig:
    hidden_size: int = MISSING
    max_sequence_length: int = 512
    num_token_types: int = 2
    embedding_dropout: float = 0.0
    learn_positional_encodings: bool = False


@dataclass
class TransformerEmbeddingConfig(EmbeddingConfig):
    __target__: str = "nemo.collections.nlp.modules.common.transformer.TransformerEmbedding"


@dataclass
class EncDecNLPModelConfig:
    enc_tokenizer: TokenizerConfig = MISSING
    dec_tokenizer: TokenizerConfig = MISSING
    vocab_divisibile_by_eight: bool = True
    enc_embedding: EmbeddingConfig = MISSING
    dec_embedding: EmbeddingConfig = MISSING


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models.
    """

    def __init__(self, cfg: EncDecNLPModelConfig, trainer: Trainer = None):
        self._enc_tokenizer = None
        self._dec_tokenizer = None
        self._enc_vocab_size = None
        self._dec_vocab_size = None
        self._enc_embedding = None
        self._dec_embedding = None

    @property
    def enc_vocab_size(self):
        return self._enc_vocab_size

    @enc_vocab_size.setter
    def enc_vocab_size(self, size: int):
        self._enc_vocab_size = size

    @property
    def dec_vocab_size(self):
        return self._dec_vocab_size

    @dec_vocab_size.setter
    def dec_vocab_size(self, size: int):
        self._dec_vocab_size = size

    @property
    def enc_tokenizer(self):
        return self._enc_tokenizer

    @enc_tokenizer.setter
    def enc_tokenizer(self, tokenizer):
        self._enc_tokenizer = tokenizer

    @property
    def dec_tokenizer(self):
        return self._enc_tokenizer

    @dec_tokenizer.setter
    def dec_tokenizer(self, tokenizer):
        self._dec_tokenizer = tokenizer

    @property
    def enc_embedding(self):
        return self._enc_embedding

    @enc_embedding.setter
    def enc_embedding(self, embedding):
        self._enc_embedding = embedding

    @property
    def dec_embedding(self):
        return self._enc_embedding

    @dec_embedding.setter
    def dec_embedding(self, embedding):
        self._dec_embedding = embedding

    def setup_enc_dec_tokenizers(self, cfg: EncDecNLPModelConfig):
        self.enc_tokenizer = get_tokenizer(**asdict(cfg.enc_tokenizer))
        self.dec_tokenizer = get_tokenizer(**asdict(cfg.dec_tokenizer))
        self.enc_vocab_size = self.enc_tokenizer.vocab_size
        self.dec_vocab_size = self.dec_tokenizer.vocab_size
        # optionally make vocabulary size divisible by 8 for fast fp16 training
        if cfg.vocab_divisibile_by_eight:
            self.enc_vocab_size = 8 * math.ceil(self.enc_tokenizer.vocab_size / 8)
            self.dec_vocab_size = 8 * math.ceil(self.dec_tokenizer.vocab_size / 8)
