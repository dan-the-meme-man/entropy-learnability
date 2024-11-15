import torch
from torch import Tensor

torch.manual_seed(42)

def create_normal_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> tuple[Tensor, Tensor]:
    
    """
        Create a bigram table with roughly normally distributed probabilities.
        
        Args:
            `vocab_size: int` - the size of the vocabulary.
            `softmax: bool` - whether to apply softmax to the probabilities.
            
        Returns:
            `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
    """
    
    bigram_probs = torch.randn(vocab_size, vocab_size)
    
    # transition to start symbol (0) with probability 0
    bigram_probs[:, 0] = 0
    
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
    
    return bigram_probs



def create_uneven_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> tuple[Tensor, Tensor]:
    
    """
        Create a bigram table with uneven probabilities.
        
        Args:
            `vocab_size: int` - the size of the vocabulary.
            `softmax: bool` - whether to apply softmax to the probabilities.
            
        Returns:
            `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
    """
    
    # randomly add a large value to a few of the bigram probabilities
    num_probs_to_change = 1 + int(0.05 * (vocab_size - 2))
    
    bigram_probs = torch.randn(vocab_size - 2, vocab_size - 2)
    
    row_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    col_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    
    for i, j in zip(row_indices_to_change, col_indices_to_change):
        bigram_probs[i, j] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
        
    # transition to start symbol (0) with probability 0
    bigram_probs[0] = 0
    
    if softmax:
        bigram_probs = torch.nn.functional.softmax(bigram_probs, dim=-1)
    else:
        bigram_probs += bigram_probs.min()
        bigram_probs = bigram_probs.abs()
        bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
    bigram_probs.requires_grad = False
    
    return bigram_probs



# def create_normal_bigram_table(
#     vocab_size: int,
#     softmax: bool = False
# ) -> tuple[Tensor, Tensor]:
    
#     """
#         Create a bigram table with roughly normally distributed probabilities.
        
#         Args:
#             `vocab_size: int` - the size of the vocabulary.
#             `softmax: bool` - whether to apply softmax to the probabilities.
            
#         Returns:
#             `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
#     """
    
#     bigram_probs = torch.randn(vocab_size - 2, vocab_size - 2)
    
#     if softmax:
#         bigram_probs = torch.nn.functional.softmax(
#             bigram_probs,
#             dim=-1
#         )
#     else:
#         bigram_probs += bigram_probs.min()
#         bigram_probs = bigram_probs.abs()
#         bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
#     start_probs = torch.randn(vocab_size - 2)
    
#     if softmax:
#         start_probs = torch.nn.functional.softmax(
#             start_probs,
#             dim=-1
#         )
#     else:
#         start_probs += start_probs.min()
#         start_probs = start_probs.abs()
#         start_probs = start_probs / start_probs.sum()
    
#     bigram_probs.requires_grad = False
#     start_probs.requires_grad = False
    
#     return bigram_probs, start_probs



# def create_uneven_bigram_table(
#     vocab_size: int,
#     softmax: bool = False
# ) -> tuple[Tensor, Tensor]:
    
#     """
#         Create a bigram table with uneven probabilities.
        
#         Args:
#             `vocab_size: int` - the size of the vocabulary.
#             `softmax: bool` - whether to apply softmax to the probabilities.
            
#         Returns:
#             `tuple[Tensor, Tensor]` - Bigram probabilities, start probabilities.
#     """
    
#     # randomly add a large value to a few of the bigram probabilities
#     num_probs_to_change = 1 + int(0.05 * (vocab_size - 2))
    
#     bigram_probs = torch.randn(vocab_size - 2, vocab_size - 2)
    
#     row_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
#     col_indices_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    
#     for i, j in zip(row_indices_to_change, col_indices_to_change):
#         bigram_probs[i, j] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
    
#     if softmax:
#         bigram_probs = torch.nn.functional.softmax(bigram_probs, dim=-1)
#     else:
#         bigram_probs += bigram_probs.min()
#         bigram_probs = bigram_probs.abs()
#         bigram_probs = bigram_probs / bigram_probs.sum(dim=-1, keepdim=True)
    
#     start_probs_to_change = torch.randint(0, vocab_size - 2, (num_probs_to_change**2,))
    
#     start_probs = torch.randn(vocab_size - 2)
    
#     for i in start_probs_to_change:
#         start_probs[i] += vocab_size - 2 + torch.randn(1).item() * 0.5 * vocab_size
    
#     if softmax:
#         start_probs = torch.nn.functional.softmax(start_probs, dim=-1)
#     else:
#         start_probs += start_probs.min()
#         start_probs = start_probs.abs()
#         start_probs = start_probs / start_probs.sum()
    
#     bigram_probs.requires_grad = False
#     start_probs.requires_grad = False
    
#     return bigram_probs, start_probs