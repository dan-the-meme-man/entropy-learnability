import argparse
from multiprocessing import freeze_support
import torch

from model import *
from generate_unigram_sequences import *
from generate_bigram_sequences import *
from unigram_tables import *
from bigram_tables import *
from train_model import *

torch.manual_seed(42)

def main():

    distributions = [
        'uniform_unigrams',
        'normal_unigrams',
        'normal_bigrams',
        'uneven_bigrams'
    ]

    vocab_sizes = [
        10,
        100,
        1000,
        10000
    ]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--distribution', '-d', type=str, choices=distributions)
    argparser.add_argument('--vocab_size', '-v', type=int, choices=vocab_sizes)
    argparser.add_argument('--softmax', '-s', type=bool, default=False)
    args = argparser.parse_args()

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
        'epochs': 5,
        'learning_rate': 0.001,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'dist': args.distribution,
        'num_train_samples': 900,
        'num_val_samples': 100
    }

    save_name = f'{hparams["dist"]}_{hparams["vocab_size"]}'
    save_name += f'_{"softmax" if args.softmax else "nosoftmax"}'
    os.makedirs('pmfs', exist_ok=True)
    
    if hparams['dist'] == 'uniform_unigrams':
        unigram_probs = create_uniform_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )

        torch.save(
            unigram_probs,
            os.path.join('pmfs', f'{save_name}.pt')
        )
        
        def get_sequences():
            return generate_unigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                unigram_probs
            )
    elif hparams['dist'] == 'normal_unigrams':
        unigram_probs = create_normal_unigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        torch.save(
            unigram_probs,
            os.path.join('pmfs', f'{save_name}.pt')
        )
        
        def get_sequences():
            return generate_unigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                unigram_probs
            )
    elif hparams['dist'] == 'normal_bigrams':
        bigram_probs, start_probs = create_normal_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        torch.save(
            bigram_probs,
            os.path.join('pmfs', f'{save_name}.pt')
        )
        torch.save(
            start_probs,
            os.path.join('pmfs', f'{save_name}_start.pt')
        )
        
        def get_sequences():
            return generate_bigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                bigram_probs,
                start_probs
            )
    elif hparams['dist'] == 'uneven_bigrams':
        bigram_probs, start_probs = create_uneven_bigram_table(
            hparams['vocab_size'],
            softmax=args.softmax
        )
        
        torch.save(
            bigram_probs,
            os.path.join('pmfs', f'{save_name}.pt')
        )
        torch.save(
            start_probs,
            os.path.join('pmfs', f'{save_name}_start.pt')
        )
        
        def get_sequences():
            return generate_bigram_sequences_using_table(
                hparams['batch_size'],
                hparams['sequence_length'],
                bigram_probs,
                start_probs
            )
    else:
        raise ValueError('Invalid distribution. Options are: ' + ', '.join(distributions))

    train_loader = DataLoader(
        [get_sequences() for _ in range(hparams['num_train_samples'])],
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        [get_sequences() for _ in range(hparams['num_val_samples'])],
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    model=get_model(**hparams)
    optimizer = get_optimizer(model, hparams)

    train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        epochs=hparams['epochs'],
        log_interval=10,
        save_name=save_name,
        scheduler=get_scheduler(optimizer, hparams['warmup_steps']),
        device=hparams['device']
    )
    
if __name__ == '__main__':
    freeze_support()
    main()