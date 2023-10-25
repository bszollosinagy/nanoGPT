import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_pmf(decode, pmf_at_each_step: torch.Tensor, gen_seq, probability_amplification_factor=2):
    gen_seq = gen_seq.detach().cpu().numpy()
    assert gen_seq.shape[0] == 1
    gen_seq = gen_seq[0]
    # Shift by 1, because start token is included in "gen_seq" but not in "pmf_at_each_step"
    gen_seq = gen_seq[1:]
    pmf_at_each_step = pmf_at_each_step.detach().cpu().numpy().T
    pmf_at_each_step = pmf_at_each_step[:, :60] ** (1/probability_amplification_factor)

    def make_printable(x):
        return '\\n' if x=='\n' else x

    decode_printable = lambda x: make_printable(decode([x]))

    fig, ax = plt.subplots(figsize=(16,16))

    im = ax.imshow(pmf_at_each_step, cmap='Blues', interpolation='none')

    for i in range(pmf_at_each_step.shape[0]):
        for j in range(pmf_at_each_step.shape[1]):
            ax.text(j, i, decode_printable(i), ha="center", va='center', color='gray')
            actually_generated_token_idx = gen_seq[j]
            ax.add_patch(Rectangle((j - .5, actually_generated_token_idx - .5), 1, 1, facecolor="none", ec='r', lw=1 ))

    plt.axis('off')
    ax.set_title("Probability Mass Function over tokens (vertically) per generation step (horizontally)")
    fig.tight_layout()

    plt.show()



