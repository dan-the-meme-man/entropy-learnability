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
from entropy import (
    calculate_entropy_unigram,
    calculate_entropy_bigram,
    calculate_transient_entropy_bigram,
    calculate_transient_entropy_unigram
)

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
        'uneven_bigrams',
        'manual_unigrams',
        'manual_bigrams'
    ]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--distribution', '-d', type=str, choices=distributions)
    argparser.add_argument('--vocab_size', '-v', type=int, default=-1)
    argparser.add_argument('--softmax', '-s', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--lstm', '-l', action='store_true')
    argparser.add_argument('--manual_option', '-m', type=float, default=0.0)
    args = argparser.parse_args()
    if args.manual_option  == 0.0 and args.distribution in ['manual_unigrams', 'manual_bigrams']:
        raise ValueError('Please provide a manual option for the distribution.')
    if args.manual_option != 0.0 and args.distribution in ['manual_unigrams', 'manual_bigrams']:
        args.softmax = False
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    hparams = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vocab_size': int(args.vocab_size),
        'n_positions': 64,
        'n_embd': 64, # 64, 256
        'n_layer': 4,
        'n_head': 4,
        'resid_pdrop': 0.05,
        'embd_pdrop': 0.05,
        'attn_pdrop': 0.05,
        'summary_first_dropout': 0.05,
        'bos_token_id': 0,
        'eos_token_id': 1,
        'pad_token_id': 2,
        'batch_size': 4,
        'sequence_length': 64,
        'epochs': 4,
        'learning_rate': 0.001,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'dist': args.distribution,
        'num_train_samples': 8000,
        'num_test_samples': 2000,
        'log_interval': 10,
        'manual_option': args.manual_option
    }
    
    if hparams['dist'] == 'uniform_unigrams':
        probs = create_uniform_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
    elif hparams['dist'] == 'normal_unigrams':
        probs = create_normal_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
    elif hparams['dist'] == 'normal_bigrams':
        probs = create_normal_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
    elif hparams['dist'] == 'uneven_bigrams':
        probs = create_uneven_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
    elif hparams['dist'] == 'manual_unigrams':
        probs = manual_unigram_table(hparams['manual_option'])
        hparams['vocab_size'] = len(probs)  
    elif hparams['dist'] == 'manual_bigrams':
        probs = manual_bigram_table(hparams['manual_option'])
        hparams['vocab_size'] = len(probs)
    else:
        raise ValueError('Invalid distribution. Options are: ' + ', '.join(distributions))
    
    if 'unigrams' in hparams['dist']:
        entropy_calc = calculate_entropy_unigram
        transient_entropy_calc = calculate_transient_entropy_unigram
    elif 'bigrams' in hparams['dist']:
        entropy_calc = calculate_entropy_bigram
        transient_entropy_calc = calculate_transient_entropy_bigram
    else:
        raise ValueError('Invalid distribution. Options are: ' + ', '.join(distributions))
    
    def get_sequences() -> Tensor:
        return generate_unigram_sequences_using_table(
            hparams['batch_size'],
            hparams['sequence_length'],
            probs,
            hparams['bos_token_id'],
            hparams['eos_token_id'],
            hparams['pad_token_id']
        )
    
    save_name = f'{hparams["dist"]}_{hparams["vocab_size"]}'
    save_name += f'_{"softmax" if args.softmax else "nosoftmax"}'
    if args.manual_option:
        save_name += f'_manual_{args.manual_option}'
    if args.lstm:
        save_name += f'_lstm'
    if hparams['n_embd'] != 64:
        save_name += f'_embd_{hparams["n_embd"]}'
        
    print('training:', save_name)
    print('training on:', hparams['device'])
    
    if os.path.exists(os.path.join('results', save_name + '.json')):
        print(f'results/{save_name} already exists. Skipping training.')
        return

    print('Loading training data...')
    l_train = []
    for _ in range(hparams['num_train_samples']):
        l_train.append(get_sequences())
        if len(l_train) % 100 == 0:
            print(f'{len(l_train)} / {hparams["num_train_samples"]}')
    train_loader = DataLoader(
        l_train,
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )
    print('Loading test data...')
    l_test = []
    for _ in range(hparams['num_test_samples']):
        l_test.append(get_sequences())
        if len(l_test) % 100 == 0:
            print(f'{len(l_test)} / {hparams["num_test_samples"]}')
    test_loader = DataLoader(
        l_test,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    if args.lstm:
        model = get_lstm(**hparams)
    else:
        model = get_model(**hparams)
    print('param count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = get_optimizer(model, hparams)

    entropy = entropy_calc(probs)
    transient_entropy = transient_entropy_calc(
        probs,
        hparams['sequence_length'],
        hparams['batch_size'],
        hparams['bos_token_id'],
        hparams['eos_token_id'],
        hparams['pad_token_id']
    )
    
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
        entropy=entropy,
        transient_entropy=transient_entropy,
        table=probs
    )
    
if __name__ == '__main__':
    freeze_support()
    main()
