import torch
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

from generate_sequences import (
    generate_bigram_sequences_using_table,
    generate_unigram_sequences_using_table
)

TOL = 5e-6

def calculate_entropy_unigram(unigram_table: torch.Tensor) -> float:
    """
    Calculate entropy of a unigram table.
    
    Args:
        unigram_table: `torch.Tensor` - the unigram table.
        
    Returns:
        `float` - the entropy of the unigram table.
    """
    
    assert torch.isclose(unigram_table.sum(), torch.tensor(1.0))
    
    unigram_table = unigram_table.to(device)
    
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
    
    # assert that the rows sum to 1
    assert torch.allclose(bigram_table.sum(1), torch.tensor(1.0))
    
    bigram_table = bigram_table.to(device)
    
    # p(symbol)
    p = _get_stationary_distribution(bigram_table)
    
    # broadcast p(symbol) to the normalized rows of the transition matrix
    joint_probs = p[:, None] * bigram_table
    
    # -sum(p * log p)
    try:
        t = -1 * (joint_probs * joint_probs.log()).sum()
    except:
        t = -1 * (joint_probs * (joint_probs + 1e-10).log()).sum()
    
    # return scalar
    return t.item()


# TODO: Find stackoverflow post about this
def _get_stationary_distribution(bigram_table: torch.Tensor) -> torch.Tensor:
    """
    Get the stationary distribution of a bigram table.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        
    Returns:
        `torch.Tensor` - the stationary distribution.
    """
    
    bigram_table = bigram_table.to(device)
    
    # vocab size
    n = bigram_table.shape[0]
    
    # init p(symbol)
    p = torch.ones(n) / n
    
    p = p.to(device)
    
    # iterate enough times
    for _ in range(10 * n):
        
        # step towards eigenvector
        p_next = p @ bigram_table
        
        # if converged, done
        if torch.norm(p_next - p) < 1e-8:
            break
        p = p_next
    return p



def calculate_transient_entropy_unigram(
    unigram_table: torch.Tensor,
    max_length: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int
) -> float:
    """
    Calculate entropy of a unigram table.
    
    Args:
        unigram_table: `torch.Tensor` - the unigram table.
        
    Returns:
        `float` - the entropy of the unigram table.
    """
    
    assert torch.isclose(unigram_table.sum(), torch.tensor(1.0))
    
    unigram_table = unigram_table.to(device)
    
    p = sample_unigram_seqs_to_convergence(
        unigram_table,
        max_length,
        batch_size,
        bos_token_id,
        eos_token_id,
        pad_token_id
    )
    
    t = -1 * (p * p.log()).sum()
    
    return t.item()



def calculate_transient_entropy_bigram(
    bigram_table: torch.Tensor,
    max_length: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int
) -> float:
    """
    Calculate entropy of a bigram table weighted by transient state probabilities
    derived by sampling.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        max_length: `int` - maximum sequence length to generate.
        batch_size: `int` - number of sequences to generate in parallel when sampling.
        
    Returns:
        `float` - the entropy of the bigram table.
    """
    
    # assert that the rows sum to 1
    assert torch.allclose(bigram_table.sum(1), torch.tensor(1.0))
    
    bigram_table = bigram_table.to(device)
    
    # get p(symbol)
    p = sample_bigram_seqs_to_convergence(
        bigram_table,
        max_length,
        batch_size,
        bos_token_id,
        eos_token_id,
        pad_token_id
    )
    
    # broadcast copies of p
    joint_probs = p[:, None] * bigram_table
    
    # entropy = -sum(p * log p)
    try:
        t = -1 * (joint_probs * joint_probs.log()).sum()
    except:
        t = -1 * (joint_probs * (joint_probs + 1e-10).log()).sum()
    
    # return scalar
    return t.item()



def sample_unigram_seqs_to_convergence(
    unigram_table: torch.Tensor,
    max_length: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Calculate the probability of all symbols in randomly sampled data to convergence.
    
    Args:
        unigram_table: `torch.Tensor` - the unigram table.
        max_length: `int` - maximum sequence length.
        batch_size: `int` - number of sequences to generate in parallel.
    
    Returns:
        `torch.Tensor` - probabilities of each symbol.
    """
    
    # count of each symbol in the generated sequences
    counts = torch.zeros(len(unigram_table), device=device, dtype=torch.float32)
    
    # vocab size
    n = unigram_table.shape[0]
    
    # initialize p(symbol)
    p = torch.ones(n, device=device, dtype=torch.float32) / n

    # iterate plenty of times
    for i in range(1000 * n):
        counts_sum = counts.sum().clamp(min=1e-8)  # Avoid division by zero
        p_next = counts / counts_sum # current estimates of p(symbol)

        if i % 200 == 0:  # Compute norm less frequently for speedup
            
            # difference between current next estimate and current estimate
            norm = torch.norm(p_next - p)
            print(f"Iteration {i}, Norm: {norm.item():.6f}")
            print(p)
            if norm < TOL: # TODO: set more reasonable threshold - maybe 2e-5
                break
        p = p_next

        # Generate unigram sequences in larger batches to better utilize GPU
        seqs = generate_unigram_sequences_using_table(
            batch_size,
            max_length,
            unigram_table,
            bos_token_id,
            eos_token_id,
            pad_token_id
        )

        # Ensure seqs is on the same device before calling `.unique`
        seqs = seqs.to(device)
        
        # get counts of unique symbols
        uc, uc_counts = seqs.unique(return_counts=True)
        
        # ignore pad token
        mask = (uc != pad_token_id)
        mask &= (uc != eos_token_id)
        mask &= (uc != bos_token_id)
        uc = uc[mask]
        uc_counts = uc_counts[mask]

        # GPU-optimized scatter_add_
        # add to running counts
        # generated indices are in range(3, n + 3)
        counts.scatter_add_(0, uc - 3, uc_counts.to(counts.dtype))

    return p



def sample_bigram_seqs_to_convergence(
    bigram_table: torch.Tensor,
    max_length: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int
) -> torch.Tensor:
    
    """
    Calculate the probability of all symbols in randomly sampled data to convergence.
    
    Args:
        bigram_table: `torch.Tensor` - the bigram table.
        max_length: `int` - maximum sequence length.
        batch_size: `int` - number of sequences to generate in parallel.
        
    Returns:
        `torch.Tensor` - probabilities of each symbol.
    """
    
    # count of each symbol in the generated sequences
    counts = torch.zeros(len(bigram_table), device=device, dtype=torch.float32)
    
    # vocab size
    n = bigram_table.shape[0]
    
    # initialize p(symbol)
    p = torch.ones(n, device=device, dtype=torch.float32) / n

    # iterate plenty of times
    for i in range(1000 * n):
        counts_sum = counts.sum().clamp(min=1e-8)  # Avoid division by zero
        p_next = counts / counts_sum # current estimates of p(symbol)

        if i % 200 == 0:  # Compute norm less frequently for speedup
            
            # difference between current next estimate and current estimate
            norm = torch.norm(p_next - p)
            print(f"Iteration {i}, Norm: {norm.item():.6f}")
            if norm < TOL: # TODO: set more reasonable threshold - maybe 2e-5
                break
        p = p_next

        # Generate bigram sequences in larger batches to better utilize GPU
        seqs = generate_bigram_sequences_using_table(
            batch_size,
            max_length,
            bigram_table,
            bos_token_id,
            eos_token_id,
            pad_token_id
        )

        # Ensure seqs is on the same device before calling `.unique`
        seqs = seqs.to(device)
        
        # get counts of unique symbols
        uc, uc_counts = seqs.unique(return_counts=True)
        
        # ignore control symbols
        mask = (uc != pad_token_id)
        mask &= (uc != bos_token_id)
        mask &= (uc != eos_token_id)
        uc = uc[mask]
        uc_counts = uc_counts[mask]

        # GPU-optimized scatter_add_
        # add to running counts
        # generated indices are in range(3, n + 3)
        counts.scatter_add_(0, uc - 3, uc_counts.to(counts.dtype))

    return p
        