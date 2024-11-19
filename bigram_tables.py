import torch
from torch import Tensor, tensor, isclose

torch.manual_seed(42)

def create_normal_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    """
        Create a bigram table with roughly normally distributed probabilities.
        
        Args:
            `vocab_size: int` - the size of the vocabulary.
            `softmax: bool` - whether to apply softmax to the probabilities.
            
        Returns:
            `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
    """
    
    bigram_probs = torch.randn(vocab_size, vocab_size)
    
    if softmax:
        bigram_probs = torch.nn.functional.softmax(
            bigram_probs,
            dim=-1
        )
    else:
        bigram_probs += bigram_probs.min()
        bigram_probs = bigram_probs.abs()
        bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
    bigram_probs.requires_grad = False
    
    assert isclose(bigram_probs.sum(1), tensor(1.0)).all()
    
    return bigram_probs



def create_uneven_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    # TODO: FIX
    
    """
        Create a bigram table with uneven probabilities.
        
        Args:
            `vocab_size: int` - the size of the vocabulary.
            `softmax: bool` - whether to apply softmax to the probabilities.
            
        Returns:
            `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
    """
    
    # randomly add a large value to a few of the bigram probabilities
    num_probs_to_change = 1 + int(0.05 * (vocab_size))
    
    bigram_probs = torch.randn(vocab_size, vocab_size)
    
    row_indices_to_change = torch.randint(0, vocab_size, (num_probs_to_change**2,))
    col_indices_to_change = torch.randint(0, vocab_size, (num_probs_to_change**2,))
    
    for i, j in zip(row_indices_to_change, col_indices_to_change):
        bigram_probs[i, j] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
    
    if softmax:
        bigram_probs = torch.nn.functional.softmax(
            bigram_probs,
            dim=-1
        )
    else:
        bigram_probs += bigram_probs.min()
        bigram_probs = bigram_probs.abs()
        bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
    bigram_probs.requires_grad = False
    
    assert isclose(bigram_probs.sum(1), tensor(1.0)).all()
    
    return bigram_probs
