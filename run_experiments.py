import os
from pathlib import Path

import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
import tqdm

from testbed import (
    DigitsClassificationProblem,
    DigitsDenoisingProblem,
    SyntheticProblem,
)
from methods import ESEstimator


def run_digits_classification(
    sigma=1, estimate_sigma_alpha=0.1, test_qty=1e100, normalize=False, prefix=""
):
    # problem generation
    dirname = f"results/digits_classification_problem/{prefix}sigma_{sigma}_sigmaalpha_{estimate_sigma_alpha}_qty_{test_qty}_normalize_{normalize}"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    dcp = DigitsClassificationProblem()
    X, Y, labels_train, labels_test = dcp.get_data(
        test_qty=test_qty, normalize=normalize
    )

    # estimate knowing sigma
    n, M = X.shape[0], X.shape[1]
    thetas_es = np.empty((len(labels_test), M, 1))
    es = ESEstimator(sigma=sigma, estimate_sigma_alpha=estimate_sigma_alpha, method='binary')
    X_ind = X.squeeze()  # X is always the same
    for ind in tqdm.tqdm(range(len(labels_test) - 1)):
        Y_ind = Y[ind].reshape(-1, 1)
        thetas_es[ind] = es.estimate_from_data(X_ind, Y_ind)
    Y_ind = Y[-1].reshape(-1, 1)  # last test image
    thetas_es[-1], traj, p = es.estimate_from_data(X_ind, Y_ind, save_traj=True)
    # compute errors
    errors_f = np.empty(len(labels_test))
    label_counts = np.empty((len(labels_test), 10))
    weighted_counts = np.empty((len(labels_test), 10))
    one_hot_labels_train = OneHotEncoder(sparse=False).fit_transform(
        labels_train.reshape(-1, 1)
    )
    for ind in tqdm.tqdm(range(len(labels_test))):
        errors_f[ind] = np.mean((X_ind @ thetas_es[ind] - Y[ind].reshape(-1, 1)) ** 2)
        label_counts[ind] = np.histogram(
            labels_train[(thetas_es[ind] > 1 / n).squeeze()], bins=np.arange(11) - 0.5
        )[0]
        weighted_counts[ind] = thetas_es[ind].squeeze() @ one_hot_labels_train
    predictions_voting = np.argmax(label_counts, axis=1)
    predictions_weighted_voting = np.argmax(weighted_counts, axis=1)

    with open(os.path.join(dirname, 'statistics.txt'), 'w') as f:
        f.write(
        f"Classification report for voting classifier:\n"
        f"{sklearn.metrics.classification_report(labels_test, predictions_voting)}\n"
    )
        f.write(
        f"Classification report for weighted classifier:\n"
        f"{sklearn.metrics.classification_report(labels_test, predictions_weighted_voting)}\n"
    )


def run_digits_denoising(n_exps, sigma, obs_index=None, normalize=True, prefix=""):
    # problem generation
    dirname = f"results/digits_problem/{prefix}n_exps_{n_exps}_sigma_{sigma}_obs_index_{obs_index}_normalize_{normalize}"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    dp = DigitsDenoisingProblem(sigma=sigma)
    X, Y, true_digit, label, labels = dp.get_data(
        n_exps=n_exps, obs_index=obs_index, normalize=normalize
    )

    # estimate knowing sigma
    n, M = X.shape[1], X.shape[2]  # X shape is (1, n, M)
    thetas_es = np.empty((n_exps, M, 1))
    es = ESEstimator(sigma=sigma)
    X_ind = X.squeeze()  # X is always the same
    for ind in tqdm.tqdm(range(n_exps - 1)):
        Y_ind = Y[ind]
        thetas_es[ind] = es.estimate_from_data(X_ind, Y_ind)
    Y_ind = Y[n_exps - 1]  # last experiment
    thetas_es[n_exps - 1], traj, p = es.estimate_from_data(X_ind, Y_ind, save_traj=True)

    # compute errors
    errors_f = np.empty(n_exps)
    errors_theta = np.empty(n_exps)
    errors_selection = np.empty(n_exps)
    label_counts = np.empty((n_exps, 10))
    for ind in tqdm.tqdm(range(n_exps)):
        error_f = np.mean((X_ind @ thetas_es[ind] - true_digit) ** 2)
        errors_f[ind] = error_f
        label_counts[ind] = np.histogram(
            labels[(thetas_es[ind] > 1 / n).squeeze()], bins=np.arange(11) - 0.5
        )[0]
    predictions = np.argmax(label_counts, axis=1)

    # visualize
    plt.figure()
    ax = plt.gca()
    plt.title(r"Labels count")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.bar(
        np.arange(10), np.eye(10)[label], width=1, color=(1, 0, 0, 0.2), label="true",
    )
    plt.bar(
        np.arange(10),
        np.mean(label_counts / np.sum(label_counts, axis=1)[:, None], axis=0),
        width=0.75,
        color=(0, 1, 0, 0.2),
        label="average ES proportion",
    )
    plt.bar(
        np.arange(10),
        label_counts[-1] / np.sum(label_counts[-1]),
        width=0.5,
        color=(0, 0, 1, 0.2),
        label="last ES",
    )
    plt.xlabel("label")
    plt.ylabel("label proportion")
    plt.legend()
    plt.savefig(os.path.join(dirname, "histogram"))
    plt.close("all")

    plt.figure()
    plt.title("Last experiment result")
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("real")
    ax1.axis("off")
    plt.imshow(true_digit.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("noisy")
    ax2.axis("off")
    plt.imshow(Y[-1].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("predicted")
    ax3.axis("off")
    plt.imshow((X_ind @ thetas_es[-1]).reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("absolute error")
    ax4.axis("off")
    plt.imshow(
        np.abs((X_ind @ thetas_es[-1]).reshape(8, 8) - true_digit.reshape(8, 8)),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    plt.savefig(os.path.join(dirname, "last_experiment_result"))
    plt.close("all")

    plt.figure()
    plt.title(r"$|X\theta^{ES} - X_{ref}|_2^2$")
    plt.boxplot(errors_f)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.savefig(os.path.join(dirname, "prediction_error"))
    plt.close("all")

    fig = plt.figure()
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.title("Evolution of p")
    plt.imshow(p[1:].squeeze().T, interpolation="nearest", aspect="auto")
    plt.xlabel(r"timestep $t$")
    plt.ylabel(r"entry index")
    plt.savefig(os.path.join(dirname, "p_evolution"))
    plt.close("all")

    fig = plt.figure()
    fig.supxlabel("entry index")
    fig.supylabel("timestep $t$")
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Evolution of traj")
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        top=False,  # ticks along the bottom edge are off
        bottom=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.imshow(traj.squeeze().T, interpolation="nearest", aspect="auto")
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Clipped evolution of traj")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.imshow(traj.squeeze().T, interpolation="nearest", vmin=0, vmax=1, aspect="auto")
    plt.savefig(os.path.join(dirname, "traj_evolution"))
    plt.close("all")


def run_synthetic(n_exps, n, M, S, X_type, prefix=""):
    # problem generation
    dirname = f"results/synthetic_problem/{prefix}n_exps_{n_exps}_n_{n}_M_{M}_S_{S}_X_type_{X_type}"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    sp = SyntheticProblem(n, M, S, X_type=X_type)
    X, Y = sp.get_data(n_exps=n_exps)

    # estimate knowing sigma
    thetas_es = np.empty((n_exps, M, 1))
    es = ESEstimator(sigma=S / 9)
    for ind in tqdm.tqdm(range(n_exps - 1)):
        X_ind, Y_ind = X[ind], Y[ind]
        thetas_es[ind] = es.estimate_from_data(X_ind, Y_ind)
    X_ind, Y_ind = X[n_exps - 1], Y[n_exps - 1]  # last experiment
    thetas_es[n_exps - 1], traj, p = es.estimate_from_data(X_ind, Y_ind, save_traj=True)

    # compute errors
    errors_f = np.empty(n_exps)
    errors_theta = np.empty(n_exps)
    errors_selection = np.empty(n_exps)
    for ind in tqdm.tqdm(range(n_exps)):
        error_f = np.mean((X[ind] @ (thetas_es[ind] - sp.theta_star)) ** 2)
        error_theta = np.mean((thetas_es[ind] - sp.theta_star) ** 2)
        selection_term1 = np.sum(
            1 / n < (thetas_es[ind][(1 - sp.theta_star).astype(bool)])
        )  # entries that should be 0 but are not
        selection_term2 = np.sum(
            thetas_es[ind][sp.theta_star.astype(bool)] < 1 / n
        )  # entries that shouldn't be 0 but are
        error_selection = (selection_term1 + selection_term2) / M

        errors_f[ind] = error_f
        errors_theta[ind] = error_theta
        errors_selection[ind] = error_selection

    # visualize
    plt.figure()
    ax = plt.gca()
    plt.title(r"Last experimient, first $2S$ coefficients")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.bar(
        np.arange(2 * S),
        sp.theta_star.flatten()[: 2 * S],
        width=1,
        color=(1, 0, 0, 0.2),
        label="true",
    )
    plt.bar(
        np.arange(2 * S),
        thetas_es[-1].flatten()[: 2 * S],
        width=0.5,
        color=(0, 0, 1, 0.2),
        label="ES",
    )
    plt.xlabel("entry index")
    plt.ylabel("entry value")
    plt.legend()
    plt.savefig(os.path.join(dirname, "last_experiment_coefficients"))
    plt.close("all")

    plt.figure()
    plt.title(r"$|X(\theta^{ES} - \theta^\star)|_2^2$")
    plt.boxplot(errors_f)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.savefig(os.path.join(dirname, "prediction_error"))
    plt.close("all")

    plt.figure()
    plt.title(r"$|\theta^{ES} - \theta^\star|_2^2$")
    plt.boxplot(errors_theta)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.savefig(os.path.join(dirname, "theta_error"))
    plt.close("all")

    plt.figure()
    plt.title(r"Percentage of erroneous selections")
    plt.boxplot(errors_selection * 100)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.savefig(os.path.join(dirname, "selection_error"))
    plt.close("all")

    fig = plt.figure()
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.title("Evolution of p")
    plt.imshow(p[1:].squeeze().T, interpolation="nearest", aspect="auto")
    plt.xlabel(r"timestep $t$")
    plt.ylabel(r"entry index")
    plt.savefig(os.path.join(dirname, "p_evolution"))
    plt.close("all")

    fig = plt.figure()
    fig.supxlabel("entry index")
    fig.supylabel("timestep $t$")
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Evolution of traj")
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        top=False,  # ticks along the bottom edge are off
        bottom=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.imshow(traj.squeeze().T, interpolation="nearest", aspect="auto")
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("Clipped evolution of traj")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.imshow(traj.squeeze().T, interpolation="nearest", vmin=0, vmax=1, aspect="auto")
    plt.savefig(os.path.join(dirname, "traj_evolution"))
    plt.close("all")


if __name__ == "__main__":
    # prefix = 'run_1_'  # knowing sigma
    # n_exps, n, M, S, X_type = 500, 100, 200, 10, 'gaussian'
    # run_synthetic(n_exps, n, M, S, X_type, prefix=prefix)

    # n_exps, n, M, S, X_type = 500, 100, 200, 10, 'rademacher'
    # run_synthetic(n_exps, n, M, S, X_type, prefix=prefix)

    # n_exps, n, M, S, X_type = 500, 200, 500, 20, 'gaussian'
    # run_synthetic(n_exps, n, M, S, X_type, prefix=prefix)

    # n_exps, n, M, S, X_type = 500, 200, 500, 20, 'rademacher'
    # run_synthetic(n_exps, n, M, S, X_type, prefix=prefix)

    # prefix = 'try2_'  # knowing sigma
    # n_exps, sigma, obs_index, normalize = 5, 1, 10, False
    # run_digits_denoising(n_exps, sigma, obs_index=obs_index, prefix=prefix, normalize=normalize)

    # sigma, estimate_sigma_alpha, test_qty, normalize = 1, None, 1000, False
    # run_digits_classification(
    #     sigma=sigma, estimate_sigma_alpha=estimate_sigma_alpha, test_qty=test_qty, normalize=normalize, prefix=prefix
    # )
    prefix = "run_1_binary_"
    sigma, estimate_sigma_alpha, test_qty, normalize = 5, 0.1, 1000, False
    run_digits_classification(
        sigma=sigma, estimate_sigma_alpha=estimate_sigma_alpha, test_qty=test_qty, normalize=normalize, prefix=prefix
    )



