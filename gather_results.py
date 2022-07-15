from glob import glob
import itertools

#################################
# Author: George P. Prodan      #
# Last modified: 7 july 2022   #
#################################

results = glob('results/seq2/*ltc_32*')
results.sort()

pars_dict = {
    "model": ["lstm", "ltc", "ltc_rk", "ltc_ex", "node", "ctgru", "ctrnn"],
    "size": [4, 8, 16, 32, 64],
    "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "opt": ['Adam', 'Adagrad', 'RMSProp', 'GD', 'Adadelta'],
    "activation": ['sigmoid', 'tanh', 'relu']
}

fixed_pars = ["model", "size"]
pars = list(pars_dict.keys())
if fixed_pars is not None:
    for par in fixed_pars:
        pars.remove(par)
confs = list(itertools.product(*[pars_dict[par] for par in pars]))

g = open('gridsearch_1.csv', 'w')
g.write("model, size, lr, opt, activation, best epoch, train loss, train accuracy, valid loss, valid accuracy, "
        "test loss, test accuracy\n")

n = 0
for r in results:
    if 'epochs' not in r:
        rf = open(r, 'r')
        line = rf.readline()
        g.write(f"ltc,32,{confs[n][0]},{confs[n][1]},{confs[n][2]}," + line)
        n += 1
        rf.close()
g.close()