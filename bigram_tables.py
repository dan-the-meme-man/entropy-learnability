import torch
from torch import Tensor

torch.manual_seed(42)

def create_normal_bigram_table(vocab_size: int, softmax=False) -> tuple[Tensor, Tensor]:
    
    bigram_probs = torch.randn(vocab_size - 2, vocab_size - 2)
    
    if softmax:
        bigram_probs = torch.nn.functional.softmax(
            bigram_probs,
            dim=-1
        )
    else:
        bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
    start_probs = torch.randn(vocab_size - 2)
    
    if softmax:
        start_probs = torch.nn.functional.softmax(
            start_probs,
            dim=-1
        )
    else:
        start_probs = start_probs / start_probs.sum()
    
    bigram_probs.requires_grad = False
    start_probs.requires_grad = False
    
    return bigram_probs, start_probs

def create_uneven_bigram_table(vocab_size: int, softmax=False) -> tuple[Tensor, Tensor]:
    
    # randomly add a large value to a few of the bigram probabilities
    num_probs_to_change = 1 + int(0.05 * (vocab_size - 2))
    
    bigram_probs = torch.randn(vocab_size - 2, vocab_size - 2)
    
    row_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    col_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    
    for i, j in zip(row_indices_to_change, col_indices_to_change):
        bigram_probs[i, j] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
    
    if softmax:
        bigram_probs = torch.nn.functional.softmax(bigram_probs, dim=-1)
    else:
        bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
    start_probs_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    
    start_probs = torch.randn(vocab_size - 2)
    
    for i in start_probs_to_change:
        start_probs[i] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
    
    if softmax:
        start_probs = torch.nn.functional.softmax(start_probs, dim=-1)
    else:
        start_probs = start_probs / start_probs.sum()
    
    bigram_probs.requires_grad = False
    start_probs.requires_grad = False
    
    return bigram_probs, start_probs