import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from lstm import LSTMLMHeadModel

def get_model(**kwargs) -> GPT2LMHeadModel:
    
    """
        Untrained GPT2 model with a custom configuration.
        Note that sequences generated for GPT2 should not include 0 or 1.
        
        Args:
            vocab_size: `int` - the size of the vocabulary.
            n_positions: `int` - the maximum length of the sequence.
            n_embd: `int` - the dimension of the embeddings.
            n_layer: `int` - the number of layers in the model.
            n_head: `int` - the number of heads in the multi-head attention mechanism.
            resid_pdrop: `float` - the dropout probability for the residual connections.
            embd_pdrop: `float` - the dropout probability for the embeddings.
            attn_pdrop: `float` - the dropout probability for the attention mechanism.
            summary_first_dropout: `float` - the dropout probability for the first token in the summary.
            bos_token_id: `int` - the beginning of sequence token.
            eos_token_id: `int` - the end of sequence token.
            device: `torch.device` - the device to use for the model.
            
        Returns:
            `GPT2LMHeadModel` - the untrained GPT2 model.
    """
    
    config = GPT2Config(
        vocab_size=kwargs['vocab_size'],
        n_positions=kwargs['n_positions'],
        n_embd=kwargs['n_embd'],
        n_layer=kwargs['n_layer'],
        n_head=kwargs['n_head'],
        resid_pdrop=kwargs['resid_pdrop'],
        embd_pdrop=kwargs['embd_pdrop'],
        attn_pdrop=kwargs['attn_pdrop'],
        summary_first_dropout=kwargs['summary_first_dropout'],
        bos_token_id=kwargs['bos_token_id'],
        eos_token_id=kwargs['eos_token_id'],
        pad_token_id=kwargs['pad_token_id'],
    )
    
    return GPT2LMHeadModel(config).to(kwargs['device'])



def get_optimizer(
    model: nn.Module,
    hparams: dict
) -> torch.optim.AdamW:
    
    """
        AdamW optimizer with custom learning rate, epsilon, and weight decay.
        
        Args:
            model: `nn.Module` - the model to optimize.
            hparams: `dict` - the hyperparameters for the optimizer.
            
        Returns:
            `torch.optim.AdamW` - the optimizer.
    """
    
    return torch.optim.AdamW(
        model.parameters(),
        lr=hparams['learning_rate'],
        eps=hparams['adam_epsilon'],
        weight_decay=hparams['weight_decay']
    )



def get_scheduler(
    optimizer: torch.optim.Optimizer,
    n_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    
    """
        Linear learning rate scheduler with warmup.
        
        Args:
            optimizer: `torch.optim.Optimizer` - the optimizer to adjust.
            n_steps: `int` - the number of steps to adjust the learning rate over.
        
        Returns:
            `torch.optim.lr_scheduler.LambdaLR` - the scheduler.
    """
    
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / n_steps, 1)
    )



def get_lstm(**kwargs):
    return LSTMLMHeadModel(**kwargs).to(kwargs['device'])
