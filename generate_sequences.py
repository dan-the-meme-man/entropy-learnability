import torch

def generate_unigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    unigram_probs: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    stop_states: tuple = (3,)
) -> torch.Tensor:
    
    """
        Generate random sequences of integers between 0 and vocab size.
        
        Args:
            batch_size: `int` - number of sequences to generate
            sequence_length: `sequence_length` - length of each sequence
            unigram_probs: `torch.Tensor` - probabilities of each integer in the vocabulary
            
        Returns:
            `torch.Tensor` - the generated sequences
    """
    
    # generate random integers between 0 and vocab size
    sequences = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    sequences[:, 0] = bos_token_id
    
    for i in range(batch_size):
        for j in range(sequence_length):
            # special tokens are 0, 1, and 2, so avoid them
            sampled_id = torch.multinomial(unigram_probs, 1) + 3
            if sampled_id in stop_states:
                sequences[i, j] = sampled_id
                break
            sequences[i, j] = sampled_id
        if j != sequence_length - 1:
            sequences[i, j + 1] = eos_token_id
            for k in range(j + 2, sequence_length):
                sequences[i, k] = pad_token_id
        else:
            sequences[i, j] = eos_token_id
            
    sequences.requires_grad = False
    
    return sequences



def generate_bigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    bigram_probs: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    stop_states: tuple = (3,)
) -> torch.Tensor:
    
    """
        Generate random sequences of integers between 0 and vocab size.
        
        Args:
            batch_size: `int` - number of sequences to generate
            sequence_length: `int` - length of each sequence
            bigram_probs: `torch.Tensor` - probabilities of each integer in the vocabulary
            
        Returns:
            `torch.Tensor` - the generated sequences
    """
    
    # generate random integers between 0 and vocab size
    sequences = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    sequences[:, 0] = bos_token_id
    
    for i in range(batch_size):
        for j in range(sequence_length):
            # special tokens are 0, 1, and 2, so avoid them
            sampled_id = torch.multinomial(bigram_probs[sequences[i, j - 1] - 3], 1) + 3
            if sampled_id in stop_states:
                sequences[i, j] = sampled_id
                break
            sequences[i, j] = sampled_id
        if j != sequence_length - 1:
            sequences[i, j+1] = eos_token_id
            for k in range(j+2, sequence_length):
                sequences[i, k] = pad_token_id
        else:
            sequences[i, j] = eos_token_id
            
    sequences.requires_grad = False
    
    # print(sequences)
    
    return sequences
