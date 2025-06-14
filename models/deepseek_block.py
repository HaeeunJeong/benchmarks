from __future__ import annotations
import torch, transformers
from ._hf_wrapper import HFWrapper
ID, _SEQ = "deepseek-ai/deepseek-R1-Distill-Qwen-1.5B", 32

def get_model(): return HFWrapper(transformers.AutoModel.from_pretrained(ID).eval())
def get_dummy_input():
    tok = transformers.AutoTokenizer.from_pretrained(ID)
    vocab = tok.vocab_size
    ids = torch.randint(0, vocab, (1, _SEQ), dtype=torch.long)
    return ids, torch.ones_like(ids)

