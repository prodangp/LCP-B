#################################
# Author: George P. Prodan      #
# Last modified: 11 july 2022   #
#################################

from subprocess import DEVNULL, STDOUT, call
import itertools
from tqdm import tqdm


# activate the local environment for LNNs
call('conda activate lct_nn', shell=True, stdout=DEVNULL, stderr=STDOUT)

class Grid:
    def __init__(self, epochs=10):
        self.pars = {
            "model": ["lstm", "node", "ctgru", "ctrnn"],
            "size": [4, 8, 16, 32, 64],
            "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            "opt": ['Adam', 'Adagrad', 'RMSProp', 'GD', 'Adadelta'],
            "activation": ['sigmoid', 'tanh', 'relu'],
            "window": [6, 12]
        }
        self.epochs = epochs

    def search(self, **kwargs):
        fixed_pars = list(kwargs.keys())
        pars = list(self.pars.keys())
        if fixed_pars is not None:
            for par in fixed_pars:
                pars.remove(par)
        for grid_pars in tqdm(list(itertools.product(*[self.pars[par] for par in pars]))):
            self.run(*list(kwargs.values()), *grid_pars)
            
    def run(self, model, size, lr, opt, activation, window):
        call(f"python 2seq_onehot_rand_w.py --model {model} --size {size} --lr {lr} --opt {opt} "
             f"--activation {activation} --window {window}", shell=True, stdout=DEVNULL, stderr=STDOUT)

grid = Grid(epochs=15)
grid.search(model='ltc', size=32, lr=0.05, opt='Adam', activation='sigmoid')
#grid.run('ltc', 8, 0.001, 'Adam', 'relu')