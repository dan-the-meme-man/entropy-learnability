import os
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    GPT2Tokenizer,
    BertTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
)

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

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

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
    'epochs': 4,
    'learning_rate': 0.001,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1.0,
    'log_interval': 100
}

for tokenizer, save_name in zip(
    [gpt2_tokenizer, bert_tokenizer, roberta_tokenizer, xlmroberta_tokenizer],
    ['gpt2', 'bert', 'roberta', 'xlmroberta']
):
    for split in range(N_FOLDS):
        
        test_data = []
        with open(f'brown_data/brown_{split}.txt', 'r', encoding='utf-8') as f:
            test_data = f.readlines()
        
        train_data = []
        for i in range(N_FOLDS):
            if i == split:
                continue
            train_data.extend(open(f'brown_data/brown_{i}.txt', 'r', encoding='utf-8').readlines())

        def collate_fn(data):
            return tokenizer(data, **settings)
        
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
        
        optimizer = get_optimizer(model, hparams)
        
        model = get_model(**hparams)
        
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
            entropy=0
        )