import torch
from torch import Tensor

torch.manual_seed(42)

def generate_unigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    unigram_probs: Tensor
) -> Tensor:
    
    """
        Generate random sequences of integers between 0 and vocab size.
    """
    
    # generate random integers between 2 and vocab size
    sequences = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    
    for i in range(batch_size):
        for j in range(1, sequence_length - 1):
            sequences[i, j] = torch.multinomial(unigram_probs, 1)
            
    sequences.requires_grad = False
    
    return sequences
