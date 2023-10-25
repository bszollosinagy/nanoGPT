import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def gini(x, w=None):
    # GINI coefficient.
    # Src: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def compute_concentration_metric(pmf_at_each_step, method_dict=None):
    spikyness = None
    if method_dict is None:
        method_dict = {'name': 'spikiness_index',
                       'probability_mass': 0.9}
    if method_dict['name'] == 'spikiness_index':
        '''
        Idea:
        Is there a metric to measure the spikiness of a probability mass function? An example measure could be: if you take the top most probable outcomes such that 50% of the probability mass is contained, and give me the number of outcomes. If fewer number of outcomes cover the top 50% of the probability mass, then I know that the pmf is quite concentrated. What does the scientific literature say about this?
        
        This is quite intuitive and can be useful in many scenarios. It would work as follows:

            a. Order the outcomes by their probabilities in decreasing order.

            b. Sum the probabilities starting from the top until you reach or exceed the desired total (e.g., 50%).

            c. The number of outcomes you needed to sum is your measure.
        FINDING:
            This measure correlates 99% with GINI (when threshold is 0.9 for this method).
            
        '''
        probability_mass_threshold = method_dict['probability_mass']
        outcome_count = pmf_at_each_step.shape[0]
        x = np.sort(pmf_at_each_step, axis=0)
        x = np.flipud(x)
        x = np.cumsum(x, axis=0)

        where_it_fill_enough_mass = x > probability_mass_threshold
        where_it_fill_enough_mass = np.argmax(where_it_fill_enough_mass, axis=0)
        where_it_fill_enough_mass = where_it_fill_enough_mass / where_it_fill_enough_mass.max()
        where_it_fill_enough_mass = 1 - where_it_fill_enough_mass
        spikyness = where_it_fill_enough_mass
    if method_dict['name'] == 'gini':
        spikyness = np.zeros((pmf_at_each_step.shape[1],))
        for c in range(pmf_at_each_step.shape[1]):
            spikyness[c] = gini(pmf_at_each_step[:, c])


    return spikyness

def make_printable(x):
    return '\\n' if x=='\n' else x

def plot_obviousness(decode, pmf_at_each_step: torch.Tensor, gen_seq, probability_amplification_factor=2, max_visible_token=60):
    gen_seq, pmf_at_each_step = _prepare_data(gen_seq, max_visible_token, pmf_at_each_step)
    #spikyness_mine = compute_concentration_metric(pmf_at_each_step, method_dict=None)
    spikyness_gini = compute_concentration_metric(pmf_at_each_step, method_dict={'name': 'gini'})

    # pearson = np.corrcoef(spikyness_mine, spikyness_gini)

    # Apply some monotonic non-linear transformation to highlight places of high obviousness
    spikyness = spikyness_gini ** 16
    plotstyle = 'curve'
    plotstyle = 'matrix'
    if plotstyle == 'curve':
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.plot(np.arange(spikyness.size), spikyness)
        ax.set_xticks(np.arange(spikyness.size), make_printable(decode(gen_seq.tolist())))
        plt.show()

    if plotstyle == 'matrix':
        fig, ax = plt.subplots(figsize=(16, 16))
        asimage = spikyness[None, :]
        im = ax.imshow(asimage, cmap='Blues', interpolation='none')
        for j in range(spikyness.size):
            ax.text(j, 0, make_printable(decode([gen_seq[j]])), ha="center", va='center', color='gray')
        plt.show()




def plot_pmf(decode, pmf_at_each_step: torch.Tensor, gen_seq, probability_amplification_factor=2, max_visible_token=60):
    gen_seq, pmf_at_each_step = _prepare_data(gen_seq, max_visible_token, pmf_at_each_step)

    pmf_at_each_step = pmf_at_each_step ** (1/probability_amplification_factor)
    fig, ax = plt.subplots(figsize=(16,16))
    im = ax.imshow(pmf_at_each_step, cmap='Blues', interpolation='none')
    for i in range(pmf_at_each_step.shape[0]):
        for j in range(pmf_at_each_step.shape[1]):
            ax.text(j, i, make_printable(decode([i])), ha="center", va='center', color='gray')
            actually_generated_token_idx = gen_seq[j]
            ax.add_patch(Rectangle((j - .5, actually_generated_token_idx - .5), 1, 1, facecolor="none", ec='r', lw=1 ))

    plt.axis('off')
    ax.set_title("Probability Mass Function over tokens (vertically) per generation step (horizontally)")
    fig.tight_layout()

    plt.show()


def _prepare_data(gen_seq, max_visible_token, pmf_at_each_step):
    gen_seq = gen_seq.detach().cpu().numpy()
    assert gen_seq.shape[0] == 1
    gen_seq = gen_seq[0]
    # Shift by 1, because start token is included in "gen_seq" but not in "pmf_at_each_step"
    gen_seq = gen_seq[1:max_visible_token + 1]
    pmf_at_each_step = pmf_at_each_step.detach().cpu().numpy().T
    pmf_at_each_step = pmf_at_each_step[:, :max_visible_token]
    return gen_seq, pmf_at_each_step



