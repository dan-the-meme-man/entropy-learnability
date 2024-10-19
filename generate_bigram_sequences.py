import torch
from torch import Tensor

torch.manual_seed(42)

def generate_bigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    bigram_probs: Tensor,
    start_probs: Tensor
) -> Tensor:
    
    """
        Generate random sequences of integers between 2 and vocab size.
        2 is the first integer that can be used as a token, because 0 and 1 are reserved.
        Bigrams are used to generate sequences.
    """
    
    # generate random integers between 2 and vocab size
    sequences = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    sequences[:, 0] = 0
    sequences[:, -1] = 1
    
    for i in range(batch_size):
        sequences[i, 1] = torch.multinomial(start_probs, 1)
        
        for j in range(2, sequence_length - 1):
            sequences[i, j] = torch.multinomial(
                bigram_probs[sequences[i, j - 1] - 2],
                1
            )
            
    sequences.requires_grad = False
    
    return sequences
