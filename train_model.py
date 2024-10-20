import os
import gc
import json
from time import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from transformers import GPT2LMHeadModel

torch.manual_seed(42)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_and_validate(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    epochs: int,
    log_interval: int,
    save_name: str,
    scheduler: Scheduler,
    device: str
):
    
    model.to(device)
    print(f'Training {save_name} on {device}.')
    
    train_losses = []
    times = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        
        model.train()
        
        train_losses.append([])
        
        for i, batch in enumerate(train_loader):
            
            start = time()
            
            optimizer.zero_grad()
            
            input_ids = batch.squeeze(0).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_val = loss.item()
            train_losses[epoch].append(loss_val)
            times.append(time() - start)
            
            if (i+1) % log_interval == 0:
                avg_loss = sum(train_losses[epoch]) / len(train_losses[epoch])
                avg_time = sum(times) / len(times)
                print(
                    f'Batch {i+1:04}/{len(train_loader):04}, Loss: {loss_val:.3f}, Avg Loss: {avg_loss:.3f}, Avg Time: {avg_time:.3f}'
                )
                
            del input_ids, attention_mask, inputs, outputs, loss
            free_memory()
                
        model.eval()
        
        total_loss = 0
        
        print(f'Eval {epoch + 1}')
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                
                input_ids = batch.squeeze(0).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item()
                
                if (i+1) % log_interval == 0:
                    avg_loss = sum(train_losses[epoch]) / len(train_losses[epoch])
                    avg_time = sum(times) / len(times)
                    print(
                        f'Batch {i+1:04}/{len(val_loader):04}, Loss: {loss_val:.3f}, Avg Loss: {avg_loss:.3f}, Avg Time: {avg_time:.3f}'
                    )
                
        avg_loss = total_loss / len(val_loader)
                
        print(f'Average validation Loss: {avg_loss:.3}')
        val_losses.append(avg_loss)

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', save_name + '.pt'))
    with open(os.path.join('results', save_name + '.json'), 'w+', encoding='utf-8') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, f, indent=4)