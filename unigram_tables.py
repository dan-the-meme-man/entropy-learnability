import torch
from torch import Tensor

torch.manual_seed(42)

def create_normal_unigram_table(vocab_size: int, softmax=False) -> tuple[Tensor, Tensor]:
    
    unigram_probs = torch.randn(vocab_size - 2)
    
    if softmax:
        unigram_probs = torch.nn.functional.softmax(unigram_probs, dim=-1)
    else:
        unigram_probs += unigram_probs.min()
        unigram_probs = unigram_probs.abs()
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    unigram_probs.requires_grad = False
    
    return unigram_probs

def create_uniform_unigram_table(vocab_size: int, softmax=False) -> tuple[Tensor, Tensor]:
    
    unigram_probs = torch.ones(vocab_size - 2)
    
    if softmax:
        unigram_probs = torch.nn.functional.softmax(unigram_probs, dim=-1)
    else:
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    unigram_probs.requires_grad = False
    
    return unigram_probs