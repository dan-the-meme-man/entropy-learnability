import torch

def generate_unigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    unigram_probs: torch.Tensor
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
    
    for i in range(batch_size):
        for j in range(sequence_length):
            sequences[i, j] = torch.multinomial(unigram_probs, 1)
            
    sequences.requires_grad = False
    
    return sequences



def generate_bigram_sequences_using_table(
    batch_size: int,
    sequence_length: int,
    bigram_probs: torch.Tensor
) -> torch.Tensor:
    
    """
        Generate random sequences of integers between 0 and vocab size.
        
        Args:
            batch_size: `int` - number of sequences to generate
            sequence_length: `sequence_length` - length of each sequence
            bigram_probs: `torch.Tensor` - probabilities of each integer in the vocabulary
            
        Returns:
            `torch.Tensor` - the generated sequences
    """
    
    # generate random integers between 0 and vocab size
    sequences = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    
    for i in range(batch_size):
        
        for j in range(sequence_length):
            sequences[i, j] = torch.multinomial(
                bigram_probs[sequences[i, j - 1]],
                1
            )
            
    sequences.requires_grad = False
    
    return sequences
