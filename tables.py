from math import ceil, log10

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



def long_range_bigram_table(
    vocab_size: int,
    softmax: bool = False
) -> Tensor:
    
    """
        Create a long-range bigram table.
        
        Returns:
            `torch.Tensor` - Manual bigram table.    
    """
    
    bigram_probs = create_normal_bigram_table(vocab_size, softmax)
    
    indices_selected = set()
    
    p_3_4_5 = torch.tensor([0.6, 0.3, 0.1])
    
    # create a perturbation starting from 5% of the vocabulary
    num_perturbations = ceil(vocab_size / 20 / log10(vocab_size))
    for j in range(num_perturbations):
        # select 3-5 random indices not previously selected, and without replacement
        # the same index cannot be selected twice in the same call
        # pick 3-5 for num_samples, pick using p_3_4_5
        num_samples = torch.multinomial(p_3_4_5, 1).item() + 3
        assert num_samples in {3, 4, 5}
        indices = torch.multinomial(
            torch.ones(vocab_size),
            num_samples=num_samples,
            replacement=False
        ).tolist()
        # none of these indices can be previously selected
        while True:
            copy = indices.copy()
            for index in copy:
                if index in indices_selected:
                    indices.remove(index)
            if len(indices) == len(copy):
                break
            indices += torch.multinomial(
                torch.ones(vocab_size),
                num_samples=len(copy) - len(indices),
                replacement=False
            ).tolist()
        # add these indices to the set of selected indices
        indices_selected.update(indices)
        
        # let the indices be:
        
        # 3: A, B, C
        # here, A is guaranteed to go to B,
        # B will go to B with high probability,
        # and B will go to C with low probability
        # and B and C are unreachable from other indices
        if len(indices) == 3:
            A, B, C, = indices
            
            for i in range(vocab_size):
                
                # eliminate all probability mass when transitioning from A or B
                bigram_probs[A, i] = 1e-6
                bigram_probs[B, i] = 1e-6
                # from C, we can go anywhere, so no change needed
                
                # B and C should be unreachable except for the mass we are about to assign
                # A doesn't need any change, as we can reach it from anywhere
                bigram_probs[i, B] = 1e-6
                bigram_probs[i, C] = 1e-6
            
            # A to B is guaranteed
            bigram_probs[A, B] = 1.0
            
            # B to B is high probability
            bigram_probs[B, B] = 0.9
            
            # B to C is low probability
            bigram_probs[B, C] = 0.1
        
        # 4: A, B, C, D
        # here, A is guaranteed to go to B or C,
        # B and C will go to B or C with high probability,
        # and B and C will go to D with low probability
        if len(indices) == 4:
            A, B, C, D = indices
            
            for i in range(vocab_size):
                
                # eliminate all probability mass when transitioning from A, B, or C
                bigram_probs[A, i] = 1e-6
                bigram_probs[B, i] = 1e-6
                bigram_probs[C, i] = 1e-6
                # from D, we can go anywhere, so no change needed
                
                # B, C, and D should be unreachable except for the mass we are about to assign
                # A doesn't need any change, as we can reach it from anywhere
                bigram_probs[i, B] = 1e-6
                bigram_probs[i, C] = 1e-6
                bigram_probs[i, D] = 1e-6
                
            # A to B or C is guaranteed
            bigram_probs[A, B] = 0.5
            bigram_probs[A, C] = 0.5
            
            # B and C to B or C is high probability
            bigram_probs[B, B] = 0.45
            bigram_probs[B, C] = 0.45
            bigram_probs[C, B] = 0.45
            bigram_probs[C, C] = 0.45
            
            # B and C to D is low probability
            bigram_probs[B, D] = 0.05
            bigram_probs[C, D] = 0.05
        
        # 5: A, B, C, D, E
        # here, A is guaranteed to go to B,
        # B is guaranteed to go to C,
        # C will go to D with high probability,
        # D will go to D with high probability,
        # and C and D will go to E with low probability
        if len(indices) == 5:
            A, B, C, D, E = indices
            
            for i in range(vocab_size):
                
                # eliminate all probability mass when transitioning from A, B, C, or D
                bigram_probs[A, i] = 1e-6
                bigram_probs[B, i] = 1e-6
                bigram_probs[C, i] = 1e-6
                bigram_probs[D, i] = 1e-6
                # from E, we can go anywhere, so no change needed
                
                # B, C, D, and E should be unreachable except for the mass we are about to assign
                # A doesn't need any change, as we can reach it from anywhere
                bigram_probs[i, B] = 1e-6
                bigram_probs[i, C] = 1e-6
                bigram_probs[i, D] = 1e-6
                bigram_probs[i, E] = 1e-6
                
            # A to B is guaranteed
            bigram_probs[A, B] = 1.0
            
            # B to C is guaranteed
            bigram_probs[B, C] = 1.0
            
            # C to D is high probability, C to E is low probability
            bigram_probs[C, D] = 0.9
            bigram_probs[C, E] = 0.1
            
            # D to D is high probability, D to E is low probability
            bigram_probs[D, D] = 0.9
            bigram_probs[D, E] = 0.1
            
        if j % 10 == 0:
            print(f'Perturbation {j}/{num_perturbations} complete.')
    
    bigram_probs.requires_grad = False
    
    # re-normalize so bigram_probs.sum(1) = 1
    bigram_probs /= bigram_probs.sum(dim=-1, keepdim=True)
    
    assert torch.isclose(bigram_probs.sum(1), torch.tensor(1.0)).all()
    
    return bigram_probs