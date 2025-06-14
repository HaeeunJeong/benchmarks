from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper

_MODEL_ID = "bert-base-uncased"
_SEQ = 32

def get_model():
    return HFWrapper(transformers.AutoModel.from_pretrained(_MODEL_ID).eval())

def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL_ID)
    vocab = tok.vocab_size
    ids = torch.randint(0, vocab, (1, _SEQ), dtype=torch.long)
    mask = torch.ones_like(ids)
    return ids, mask

