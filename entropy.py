from torch import Tensor, ones, norm

def calculate_entropy_unigram(unigram_table: Tensor) -> float:
    """
    Calculate entropy of a unigram table.
    """
    return -1 * (unigram_table * unigram_table.log()).sum()



def calculate_entropy_bigram(bigram_table: Tensor) -> float:
    """
    Calculate entropy of a bigram table.
    """
    
    p = get_stationary_distribution(bigram_table)
    
    joint_probs = p[:, None] * bigram_table
    
    try:
        return -1 * (joint_probs * joint_probs.log()).sum()
    except:
        return -1 * (joint_probs * (joint_probs + 1e-10).log()).sum()



def get_stationary_distribution(bigram_table: Tensor) -> Tensor:
    """
    Get the stationary distribution of a bigram table.
    """
    
    n = bigram_table.shape[0]
    p = ones(n) / n
    for _ in range(10 * n):
        p_next = p @ bigram_table
        if norm(p_next - p) < 1e-8:
            break
        p = p_next
    return p