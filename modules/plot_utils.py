import numpy as np


def plot_curve(plt, ax, xs, means, stds, marker=None, label=None):
    means = np.array(means)
    stds = np.array(stds)
    ax.plot(xs, means, label=label, marker=marker)
    plt.fill_between(xs, means-stds, means+stds, alpha=0.2)
    return ax


def plot_over_n(plt, results, ns, epoch):
    fig, ax = plt.subplots(figsize=(6, 4))

    fcmi_means = []
    fcmi_stds = []
    gap_means = []
    gap_stds = []

    for n in ns:
        fcmi_mean = np.nanmean([d['fcmi_bound'] for d in results[n][epoch]])
        fcmi_std = np.nanstd([d['fcmi_bound'] for d in results[n][epoch]])
        if np.isnan(fcmi_std):
            fcmi_std = 0.0
        gap_mean = np.nanmean([d['exp_gap'] for d in results[n][epoch]])
        gap_std = np.nanstd([d['exp_gap'] for d in results[n][epoch]])
        if np.isnan(gap_std):
            gap_std = 0.0
        fcmi_means.append(fcmi_mean)
        fcmi_stds.append(fcmi_std)
        gap_means.append(gap_mean)
        gap_stds.append(gap_std)

    plot_curve(plt, ax, range(len(ns)), gap_means, gap_stds, label=f'generalization gap', marker='o')
    plot_curve(plt, ax, range(len(ns)), fcmi_means, fcmi_stds, label=f'f-CMI bound', marker='x')
    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.legend()
    fig.tight_layout()

    print(gap_means)
    print(fcmi_means)

    return fig, ax


def plot_over_epochs(plt, results, n, epochs):
    fig, ax = plt.subplots(figsize=(6, 4))

    fcmi_means = []
    fcmi_stds = []
    gap_means = []
    gap_stds = []
    for epoch in epochs:
        fcmi_mean = np.nanmean([d['fcmi_bound'] for d in results[n][epoch]])
        fcmi_std = np.nanstd([d['fcmi_bound'] for d in results[n][epoch]])
        if np.isnan(fcmi_std):
            fcmi_std = 0.0
        gap_mean = np.nanmean([d['exp_gap'] for d in results[n][epoch]])
        gap_std = np.nanstd([d['exp_gap'] for d in results[n][epoch]])
        if np.isnan(gap_std):
            gap_std = 0.0

        fcmi_means.append(fcmi_mean)
        fcmi_stds.append(fcmi_std)
        gap_means.append(gap_mean)
        gap_stds.append(gap_std)

    plot_curve(plt, ax, epochs, gap_means, gap_stds, label=f'generalization gap', marker='o')
    plot_curve(plt, ax, epochs, fcmi_means, fcmi_stds, label=f'f-CMI bound', marker='x')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend()
    fig.tight_layout()
    return fig, ax
