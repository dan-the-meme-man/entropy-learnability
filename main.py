import os
import argparse
import random
from multiprocessing import freeze_support

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from generate_sequences import *
from tables import *
from model import get_model, get_optimizer, get_scheduler, get_lstm
from train_model import train_and_test
from entropy import calculate_entropy_unigram, calculate_entropy_bigram

def main() -> None:
    
    """
        Main function for training a model on sequences generated from a given distribution.
        
        Usage:
            python main.py -d uniform_unigrams -v 100 -s
            python main.py --distribution normal_unigrams --vocab_size 1000 --softmax
            python main.py -d normal_bigrams -v 10000
            etc.
    """

    distributions = [
        'uniform_unigrams',
        'normal_unigrams',
        'normal_bigrams',
        'uneven_bigrams'
    ]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--distribution', '-d', type=str, choices=distributions)
    argparser.add_argument('--vocab_size', '-v', type=int)
    argparser.add_argument('--softmax', '-s', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lstm', '-l', action='store_true')
    args = argparser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hparams = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vocab_size': int(args.vocab_size),
        'n_positions': 64,
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
        'dist': args.distribution,
        'num_train_samples': 80,#8000,
        'num_test_samples': 20,#2000,
        'log_interval': 10
    }

    save_name = f'{hparams["dist"]}_{hparams["vocab_size"]}'
    save_name += f'_{"softmax" if args.softmax else "nosoftmax"}'
    
    if hparams['dist'] == 'uniform_unigrams':
        unigram_probs = create_uniform_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        def get_sequences() -> Tensor:
            return generate_unigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                unigram_probs
            )
            
        entropy = calculate_entropy_unigram(unigram_probs)
    elif hparams['dist'] == 'normal_unigrams':
        unigram_probs = create_normal_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        def get_sequences() -> Tensor:
            return generate_unigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                unigram_probs
            )
            
        entropy = calculate_entropy_unigram(unigram_probs)
    elif hparams['dist'] == 'normal_bigrams':
        bigram_probs = create_normal_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        def get_sequences() -> Tensor:
            return generate_bigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                bigram_probs
            )
        
        entropy = calculate_entropy_bigram(bigram_probs)
    elif hparams['dist'] == 'uneven_bigrams':
        bigram_probs = create_uneven_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        def get_sequences() -> Tensor:
            return generate_bigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                bigram_probs
            )
            
        entropy = calculate_entropy_bigram(bigram_probs)
    else:
        raise ValueError('Invalid distribution. Options are: ' + ', '.join(distributions))

    train_loader = DataLoader(
        [get_sequences() for _ in range(hparams['num_train_samples'])],
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        [get_sequences() for _ in range(hparams['num_test_samples'])],
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    if args.lstm:
        model = get_lstm(**hparams)
    else:
        model = get_model(**hparams)
    optimizer = get_optimizer(model, hparams)
    
    if args.lstm:
        save_name += f'_lstm'
        
    print('training:', save_name)
    print('training on:', hparams['device'])
    print('param count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if os.path.exists(os.path.join('results', save_name + '.json')):
        print(f'results/{save_name} already exists. Skipping training.')
        return

    train_and_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=hparams['epochs'],
        log_interval=hparams['log_interval'],
        save_name=save_name,
        scheduler=get_scheduler(optimizer, hparams['warmup_steps']),
        device=hparams['device'],
        entropy=entropy
    )
    
if __name__ == '__main__':
    freeze_support()
    main()
