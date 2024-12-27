import os
import gc
import json
from time import time

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()



def perplexity(
    model: GPT2LMHeadModel,
    test_loader: DataLoader,
    device: str,
    text_data: bool = False
) -> float:
    
    """
        Calculate the perplexity of a GPT2 model.
        
        Args:
            model: `GPT2LMHeadModel` - the model to evaluate.
            test_loader: `DataLoader` - the DataLoader for the data.
            device: `str` - the device to use for the model.
            
        Returns:
            `float` - the perplexity of the model.
    """
    
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for inputs in test_loader:
            
            if text_data:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            else:
                input_ids = inputs.squeeze(0)
                attention_mask = torch.ones_like(input_ids)
            
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            total_loss += loss.item()
            
    return torch.exp(torch.tensor(total_loss / len(test_loader))).item()



def train_and_test(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    log_interval: int,
    save_name: str,
    scheduler: _LRScheduler,
    device: str,
    entropy: float,
    text_data: bool = False
) -> None:
    
    """
        Train and validate a GPT2 model. Logs losses to file and saves the model.
        
        Args:
            model: GPT2LMHeadModel` - GPT2 model.
            train_loader: `DataLoader` - DataLoader for training data.
            val_loader: `DataLoader` - DataLoader for validation data.
            optimizer: `Optimizer` - Optimizer for training.
            epochs: `int` - Number of epochs to train for.
            log_interval: `int` - Number of batches to wait before logging training status.
            save_name: `str` - Name of the model and results file.
            scheduler: `_LRScheduler` - Learning rate scheduler.
            device: `str` - Device to train on.
    """
    
    model.to(device)
    
    train_losses = []
    perplexities = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        
        model.train()
        
        train_losses.append([])
        
        start = time()
        total_loss_this_epoch = 0.0
        
        for i, inputs in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            if text_data:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            else:
                input_ids = inputs.squeeze(0).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
            
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            train_losses[epoch].append(loss_val)
            total_loss_this_epoch += loss_val
            
            if (i+1) % log_interval == 0:
                avg_loss = total_loss_this_epoch / (i+1)
                avg_time = (time() - start) / (i+1)
                
                msg = f'Batch {i+1:05}/{len(train_loader):05}, '
                msg += f'Loss: {loss_val:.3f}, '
                msg += f'Avg Loss: {avg_loss:.3f}, '
                msg += f'Avg Time: {avg_time:.3f}'
                print(msg, flush=True)
                
        perplexities.append(perplexity(model, test_loader, device, text_data))

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', save_name + '.json'), 'w+', encoding='utf-8') as f:
        json.dump({
            'test_set_perplexities': perplexities,
            'entropy': entropy,
            'train_losses': train_losses
        }, f, indent=4)
