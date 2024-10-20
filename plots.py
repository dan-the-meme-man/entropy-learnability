import os
import json
from itertools import chain

import matplotlib.pyplot as plt

results = sorted(os.listdir('results'))

marker_and_color = {
    'normal_bigrams_10.json': ('r', 'o'),
    'normal_bigrams_100.json': ('r', 'x'),
    'normal_bigrams_1000.json': ('r', '^'),
    'normal_bigrams_10000.json': ('r', 's'),
    'normal_unigrams_10.json': ('b', 'o'),
    'normal_unigrams_100.json': ('b', 'x'),
    'normal_unigrams_1000.json': ('b', '^'),
    'normal_unigrams_10000.json': ('b', 's'),
    'uneven_bigrams_10.json': ('g', 'o'),
    'uneven_bigrams_100.json': ('g', 'x'),
    'uneven_bigrams_1000.json': ('g', '^'),
    'uneven_bigrams_10000.json': ('g', 's'),
    'uniform_unigrams_10.json': ('y', 'o'),
    'uniform_unigrams_100.json': ('y', 'x'),
    'uniform_unigrams_1000.json': ('y', '^'),
    'uniform_unigrams_10000.json': ('y', 's'),
}
        
def make_plots(results, title, fignums):
    
    fname = '_'.join(title.lower().split())
    
    plt.figure(fignums[0], figsize=(10, 10), dpi=400)
    plt.figure(fignums[1])
    
    for result in results:
        data = json.load(open(f'results/{result}'))
        color, marker = marker_and_color[result]
        print(color, marker, result)
        train = list(chain(*data['train_losses']))
        val = data['val_losses']
        plt.figure(fignums[0])
        plt.scatter(
            list(range(1, 1+len(train))),
            train,
            color=color,
            marker=marker,
            label=result[:-5],
            s=10,
            zorder=2
        )
        plt.figure(fignums[1])
        plt.scatter(
            list(range(1, 1+len(val))),
            val,
            color=color,
            marker=marker,
            label=result[:-5],
            s=20,
            zorder=2
        )
        
    plt.figure(fignums[0])
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig(f'plots/{fname}_train.png')
    
    plt.figure(fignums[1])
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel('Validation at each epoch')
    plt.ylabel('Average validation loss')
    plt.savefig(f'plots/{fname}_val.png')
    
fignums_list = [(i, i+1) for i in range(1, 2*len(results)+1, 2)]

titles = [
    'Vocabulary size 10',
    'Vocabulary size 100',
    'Vocabulary size 1000',
    'Vocabulary size 10000',
    'Normal bigrams',
    'Normal unigrams',
    'Uneven bigrams',
    'Uniform unigrams'
]

results_list = [
    [x for x in results if '10.json' in x],
    [x for x in results if '100.json' in x],
    [x for x in results if '1000.json' in x],
    [x for x in results if '10000.json' in x],
    [x for x in results if 'normal_bigrams' in x],
    [x for x in results if 'normal_unigrams' in x],
    [x for x in results if 'uneven_bigrams' in x],
    [x for x in results if 'uniform_unigrams' in x]
]
    
for fignums, title, results in zip(fignums_list, titles, results_list):
    make_plots(results, title, fignums)