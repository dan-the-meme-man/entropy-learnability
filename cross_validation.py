import os
import sys
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer

from train_model import train_and_test
from model import get_model, get_optimizer, get_scheduler

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

N_FOLDS = len(os.listdir('brown_data'))

settings = {
    'padding': 'longest',
    'max_length': 128,
    'truncation': True,
    'return_tensors': 'pt',
    'return_token_type_ids': False,
}

hparams = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_positions': settings['max_length'],
    'n_embd': 64,
    'n_layer': 4,
    'n_head': 4,
    'resid_pdrop': 0.05,
    'embd_pdrop': 0.05,
    'attn_pdrop': 0.05,
    'summary_first_dropout': 0.05,
    'bos_token_id': 0,
    'eos_token_id': 1,
    'batch_size': 4,
    'sequence_length': 64,
    'epochs': 10,
    'learning_rate': 0.001,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1.0,
    'log_interval': 100
}

save_name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(save_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
save_name = save_name.split('/')[-1].replace('-', '_')

hparams['vocab_size'] = tokenizer.vocab_size

def collate_fn(data):
    return tokenizer(data, **settings)

print('Using tokenizer:', save_name, flush=True)

for split in range(N_FOLDS):
    
    print('Split:', split, flush=True)
    
    test_data = []
    with open(f'brown_data/brown_{split}.txt', 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    train_data = []
    for i in range(N_FOLDS):
        if i == split:
            continue
        train_data.extend(open(f'brown_data/brown_{i}.txt', 'r', encoding='utf-8').readlines())
    
    train_loader = DataLoader(
        train_data,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=hparams['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model = get_model(**hparams)
    
    optimizer = get_optimizer(model, hparams)
    
    train_and_test(
        model,
        train_loader,
        test_loader,
        optimizer=optimizer,
        epochs=hparams['epochs'],
        log_interval=hparams['log_interval'],
        save_name=save_name + f'_{split}',
        scheduler=get_scheduler(optimizer, hparams['warmup_steps']),
        device=hparams['device'],
        entropy=0,
        text_data=True
    )
