import torch

from generate_sequences import generate_bigram_sequences_using_table

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


# TODO: Find stackoverflow post about this
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



def calculate_transient_entropy(
    bigram_table: torch.Tensor,
    max_length: int = 128,
    batch_size: int = 256
):
    """
    Calculate entropy of a bigram table weighted by transient state probabilities
    derived by sampling.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        max_length: `int` - maximum sequence length.
        batch_size: `int` - number of sequences to generate in parallel.
        
    Returns:
        `float` - the entropy of the bigram table.
    """
    
    p = sample_bigram_seqs_to_convergence(
        bigram_table,
        max_length,
        batch_size
    )
    
    joint_probs = p[:, None] * bigram_table
    
    try:
        t = -1 * (joint_probs * joint_probs.log()).sum()
    except:
        t = -1 * (joint_probs * (joint_probs + 1e-10).log()).sum()
        
    return t.item()


def sample_bigram_seqs_to_convergence(
    bigram_table: torch.Tensor,
    max_length: int = 128,
    batch_size: int = 256
):
    device = bigram_table.device  # gpu if possible
    counts = torch.zeros(len(bigram_table), device=device, dtype=torch.float32)
    n = bigram_table.shape[0]
    p = torch.full((n,), 1 / n, device=device, dtype=torch.float32)

    for i in range(1000 * n):
        counts_sum = counts.sum().clamp(min=1e-8)  # Avoid division by zero
        p_next = counts / counts_sum

        if i % 50 == 0:  # Compute norm less frequently for speedup
            norm = torch.norm(p_next - p)
            print(f"Iteration {i}, Norm: {norm.item():.6f}")
            if norm < 1e-8: # TODO: set more reasonable threshold
                break
        p = p_next

        # Generate bigram sequences in larger batches to better utilize GPU
        seqs = generate_bigram_sequences_using_table(
            batch_size,
            max_length,
            bigram_table
        )

        # Ensure seqs is on the same device before calling `.unique`
        seqs = seqs.to(device)
        uc, uc_counts = seqs.unique(return_counts=True)

        # GPU-optimized scatter_add_
        counts.scatter_add_(0, uc, uc_counts.to(counts.dtype))

    return p
        