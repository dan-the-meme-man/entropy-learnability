import torch
from torch import Tensor, tensor, isclose

def create_normal_unigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    """
        Create a unigram table with roughly normally distributed probabilities.
        
        Args:
            vocab_size: `int` - Size of the vocabulary.
            softmax: `bool` - Whether to apply a softmax to the probabilities.
            
        Returns:
            `Tensor` - Unigram probability table.
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
            vocab_size: `int` - Size of the vocabulary.
            softmax: `bool` - Whether to apply a softmax to the probabilities.
            This option is included for consistency with other functions, but it is not necessary.
            
        Returns:
            `Tensor` - Unigram probability table.
    """
    
    unigram_probs = torch.ones(vocab_size)
    
    if softmax:
        unigram_probs = torch.nn.functional.softmax(unigram_probs, dim=-1)
    else:
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    unigram_probs.requires_grad = False
    
    assert isclose(unigram_probs.sum(), tensor(1.0))
    
    return unigram_probs



def create_normal_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> torch.Tensor:
    
    """
        Create a bigram table with roughly normally distributed probabilities.
        
        Args:
            vocab_size: `int` - The size of the vocabulary.
            softmax: `bool` - Whether to apply softmax to the probabilities.
            
        Returns:
            `torch.Tensor` - Bigram transition probability table.
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
    
    assert torch.isclose(bigram_probs.sum(1), torch.tensor(1.0)).all()
    
    return bigram_probs



def create_uneven_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> torch.Tensor:
    
    """
        Create a bigram table with uneven probabilities.
        
        Args:
            vocab_size: `int` - The size of the vocabulary.
            softmax: `bool` - Whether to apply softmax to the probabilities.
            
        Returns:
            `torch.Tensor` - Bigram transition probability table.
    """
    
    bigram_probs = torch.randn(vocab_size, vocab_size)
    
    # randomly add a large value to a few of the bigram probabilities
    num_probs_to_change = 1 + int(0.05 * (vocab_size))
    row_indices_to_change = torch.randint(0, vocab_size, (num_probs_to_change**2,))
    col_indices_to_change = torch.randint(0, vocab_size, (num_probs_to_change**2,))
    
    for i, j in zip(row_indices_to_change, col_indices_to_change):
        bigram_probs[i, j] += vocab_size + torch.randn(1).item() * 0.5 * vocab_size
    
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
    
    assert torch.isclose(bigram_probs.sum(1), torch.tensor(1.0)).all()
    
    return bigram_probs



def create_normal_trigram_table(
    vocab_size: int,
    softmax: bool = False
) -> torch.Tensor:
    
    """
        Create a trigram table with roughly normally distributed probabilities.
        
        Args:
            vocab_size: `int` - The size of the vocabulary.
            softmax: `bool` - Whether to apply softmax to the probabilities.
            
        Returns:
            `torch.Tensor` - Trigram transition probability table.
    """
    
    trigram_probs = torch.randn(vocab_size, vocab_size, vocab_size)
    
    if softmax:
        trigram_probs = torch.nn.functional.softmax(
            trigram_probs,
            dim=-1
        )
    else:
        trigram_probs += trigram_probs.min()
        trigram_probs = trigram_probs.abs()
        trigram_probs = trigram_probs / trigram_probs.sum(dim=-1, keepdim=True)
    
    trigram_probs.requires_grad = False
    
    assert torch.isclose(trigram_probs.sum(dim=-1), torch.tensor(1.0)).all()
    
    return trigram_probs



def manual_unigram_table(manual_option: float) -> Tensor:
    
    """
        Create a manual unigram table.
        
        Returns:
            `torch.Tensor` - Manual unigram table.
    """
    
    if manual_option == 0.8:
        n = 8192
        unigram_probs = torch.tensor(
            [0.8] + [0.2 / n] * n
        )
    elif manual_option == 0.7:
        n = 282
        unigram_probs = torch.tensor(
            [0.7] + [0.3 / n] * n
        )
    elif manual_option == 0.6:
        n = 59
        unigram_probs = torch.tensor(
            [0.6] + [0.4 / n] * n
        )
    elif manual_option == 0.5:
        n = 25
        unigram_probs = torch.tensor(
            [0.5] + [0.5 / n] * n
        )
    else:
        raise ValueError('Invalid manual option.')
    
    unigram_probs.requires_grad = False
    
    assert isclose(unigram_probs.sum(), tensor(1.0))
    
    return unigram_probs



def manual_bigram_table() -> Tensor:
    
    """
        Create a manual bigram table.
        
        Returns:
            `torch.Tensor` - Manual bigram table.    
    """
    
    bigram_probs = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.1],
        [0.3, 0.2, 0.1, 0.4]
    ])
    
    bigram_probs.requires_grad = False
    
    assert torch.isclose(bigram_probs.sum(1), torch.tensor(1.0)).all()
    
    return bigram_probs