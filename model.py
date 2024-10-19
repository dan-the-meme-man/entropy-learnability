import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)

def get_model(**kwargs) -> GPT2LMHeadModel:
    
    """
        Untrained GPT2 model with a custom configuration.
        Note that sequences generated for GPT2 should not include 0 or 1.
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
        eos_token_id=kwargs['eos_token_id']
    )
    
    if torch.cuda.is_available():
        return GPT2LMHeadModel(config).cuda()
    else:
        return GPT2LMHeadModel(config)
    
def get_optimizer(model, hparams):
    
    """
        AdamW optimizer with custom learning rate, epsilon, and weight decay.
    """
    
    return torch.optim.AdamW(
        model.parameters(),
        lr=hparams['learning_rate'],
        eps=hparams['adam_epsilon'],
        weight_decay=hparams['weight_decay']
    )
    
def get_scheduler(optimizer, n_steps):
    
    """
        Linear learning rate scheduler with warmup.
    """
    
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / n_steps, 1)
    )
