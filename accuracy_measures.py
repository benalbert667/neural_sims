import numpy as np
from itertools import permutations
from math import factorial


# align states for state coverage
# also sum kl_divergences from all state matching
def state_match(model, state_probs):
    # match model.p_emissions to real state probabilities
    n_states = len(state_probs)
    best_kl = None
    best_match = None

    model_indxs = (tuple(range(n_states)) for _ in range(int(factorial(n_states))))
    sp_indxs = permutations(range(n_states))
    for ii, jj in zip(model_indxs, sp_indxs):
        kl = 0
        for i, j in zip(ii, jj):
            kl += kl_divergence(model.p_emissions[i], state_probs[j])
        if best_match is None or best_kl < kl:
            best_kl = kl
            best_match = jj

    # returns rearranged model.p_emissions array (to match real states)
    # and this assignment's total kl divergence
    return np.array(model.p_emissions)[list(best_match)], best_kl


# p = HMMs guess, distribution that is diverging
# q = generated test, "example" distribution being diverged from
def kl_divergence(p, q):
    p, q = np.array(p), np.array(q)
    return -np.sum(p * np.log(p / q))


# t = true state assignments
# p = predicted state assignments
# returns the percentage of the data correctly assigned
def state_coverage(t, p):
    t, p = np.array(t), np.array(p)
    return np.sum(t == p) / t.size


def latent_state_to_state_assignments(ls):
    # return an array of state assignments based on which state has the highest probability at each time
    # assuming ls is a 2d array of size num_states X time_steps
    return np.argmax(ls, axis=0)  # assuming states are labeled 0 to num_states-1
