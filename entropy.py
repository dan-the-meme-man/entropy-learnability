import torch

def calculate_entropy_unigram(unigram_table: torch.Tensor) -> float:
    """
    Calculate entropy of a unigram table.
    
    Args:
        unigram_table: `torch.Tensor` - the unigram table.
        
    Returns:
        `float` - the entropy of the unigram table.
    """
    
    t = -1 * (unigram_table * unigram_table.log()).sum()
    
    return t.item()



def calculate_entropy_bigram(bigram_table: torch.Tensor) -> float:
    """
    Calculate entropy of a bigram table.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        
    Returns:
        `float` - the entropy of the bigram table.
    """
    
    p = get_stationary_distribution(bigram_table)
    
    joint_probs = p[:, None] * bigram_table
    
    try:
        t = -1 * (joint_probs * joint_probs.log()).sum()
    except:
        t = -1 * (joint_probs * (joint_probs + 1e-10).log()).sum()
        
    return t.item()


# Find stackoverflow post about this
def get_stationary_distribution(bigram_table: torch.Tensor) -> torch.Tensor:
    """
    Get the stationary distribution of a bigram table.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        
    Returns:
        `torch.Tensor` - the stationary distribution.
    """
    
    n = bigram_table.shape[0]
    p = torch.ones(n) / n
    for _ in range(10 * n):
        p_next = p @ bigram_table
        if torch.norm(p_next - p) < 1e-8:
            break
        p = p_next
    return p
