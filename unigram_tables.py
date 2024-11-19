import torch
from torch import Tensor, tensor, isclose

torch.manual_seed(42)

def create_normal_unigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    """
        Create a unigram table with roughly normally distributed probabilities.
        
        Args:
            `vocab_size: int` - Size of the vocabulary.
            `softmax: bool` - Whether to apply a softmax to the probabilities.
            
        Returns:
            `Tensor` - Unigram probabilities.
    """
    
    unigram_probs = torch.randn(vocab_size)
    
    if softmax:
        unigram_probs = torch.nn.functional.softmax(unigram_probs, dim=-1)
    else:
        unigram_probs += unigram_probs.min()
        unigram_probs = unigram_probs.abs()
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    unigram_probs.requires_grad = False
    
    assert isclose(unigram_probs.sum(), tensor(1.0))
    
    return unigram_probs



def create_uniform_unigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    """
        Create a unigram table with uniform probabilities.
        
        Args:
            `vocab_size: int` - Size of the vocabulary.
            `softmax: int` - Whether to apply a softmax to the probabilities.
            This option is included for consistency with other functions, but it is not necessary.
            
        Returns:
            `Tensor` - Unigram probabilities.
    """
    
    unigram_probs = torch.ones(vocab_size)
    
    if softmax:
        unigram_probs = torch.nn.functional.softmax(unigram_probs, dim=-1)
    else:
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    unigram_probs.requires_grad = False
    
    assert isclose(unigram_probs.sum(), tensor(1.0))
    
    return unigram_probs
