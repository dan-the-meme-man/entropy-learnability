import torch
from torch import Tensor

torch.manual_seed(42)

def uniform_unigrams(batch_size: int, sequence_length: int, vocab_size: int) -> Tensor:
    
    """
        Generate random sequences of integers between 2 and vocab size.
        2 is the first integer that can be used as a token, because 0 and 1 are reserved.
        Unigrams are uniformly distributed.
    """
    
    without_special_tokens = torch.randint(2, vocab_size, (batch_size, sequence_length - 2))
    
    with_special_tokens = torch.cat([
        torch.full((batch_size, 1), 0, dtype=torch.long),
        without_special_tokens,
        torch.full((batch_size, 1), 1, dtype=torch.long)
    ], dim=1)
    
    with_special_tokens.requires_grad = False
    
    return with_special_tokens

# normally distributed unigrams, with some integers being more likely than others
def normal_unigrams(batch_size: int, sequence_length: int, vocab_size: int) -> Tensor:
    
    """
        Generate random sequences of integers between 2 and vocab size.
        2 is the first integer that can be used as a token, because 0 and 1 are reserved.
        Unigrams are normally distributed, with a preference for mid range integers.
    """
    
    # generate random floats between 2 and vocab size and round them to the nearest integer
    random_floats = torch.randn(
        batch_size,
        sequence_length - 2
    ) * (vocab_size - 1) / 2 + (vocab_size) / 2
    rounded_down = torch.floor(random_floats).long().clamp(0, vocab_size - 1)
    
    with_special_tokens = torch.cat([
        torch.full((batch_size, 1), 0, dtype=torch.long),
        rounded_down,
        torch.full((batch_size, 1), 1, dtype=torch.long)
    ], dim=1)
    
    with_special_tokens.requires_grad = False
    
    return with_special_tokens
