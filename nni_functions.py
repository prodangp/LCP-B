#################################
# Author: George P. Prodan      #
# Last modified: 10 july 2022   #
#################################


def linear(pos, k=0.5):
    return 1 - k/5 * pos


def inv(pos, k=0.5):
    return k / pos


def inv_sq(pos, k=0.5):
    return k / (pos ** 2)


def exp(pos, k=0.5):
    return k ** pos


interaction_score = {'linear': linear,
                     'inv': inv,
                     'inv_sq': inv_sq,
                     'exp': exp,
                     'None': lambda x, y: 0}
