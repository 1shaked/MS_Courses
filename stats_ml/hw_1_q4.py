import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-cache"))

try:
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


np.set_printoptions(precision=4, suppress=True)


TRUE_WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=float)
TRUE_MEANS = np.array([-1.0, 4.0, 9.0], dtype=float)
TRUE_VARIANCES = np.array([2.0, 3.0, 1.0], dtype=float)


def sample_gmm(n_samples, weights, means, variances, seed=None):
    rng = np.random.default_rng(seed)
    components = rng.choice(len(weights), size=n_samples, p=weights)
    samples = rng.normal(
        loc=means[components],
        scale=np.sqrt(variances[components]),
    )
    return samples, components


def gaussian_log_pdf_1d(x, mean, variance):
    variance = max(float(variance), 1e-8)
    return -0.5 * (np.log(2.0 * np.pi * variance) + ((x - mean) ** 2) / variance)


def robust_logsumexp(values, cutoff=-10.0):
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    mask = (values - m) > cutoff
    s = np.sum(np.exp(values[mask] - m))
    return m + np.log(s), m, s, mask


def e_step(x, weights, means, variances, cutoff=-10.0):
    n_samples = x.shape[0]
    n_components = weights.shape[0]
    responsibilities = np.zeros((n_samples, n_components), dtype=float)
    log_likelihood = 0.0

    for i in range(n_samples):
        log_terms = np.array(
            [
                np.log(max(weights[k], 1e-12))
                + gaussian_log_pdf_1d(x[i], means[k], variances[k])
                for k in range(n_components)
            ]
        )
        log_denom, m, s, mask = robust_logsumexp(log_terms, cutoff=cutoff)
        log_likelihood += log_denom
        responsibilities[i, mask] = np.exp(log_terms[mask] - m) / s

    return responsibilities, log_likelihood


def m_step(x, responsibilities):
    n_samples = x.shape[0]
    nk = responsibilities.sum(axis=0)
    weights = nk / n_samples
    means = (responsibilities * x[:, None]).sum(axis=0) / np.maximum(nk, 1e-12)
    variances = (
        responsibilities * (x[:, None] - means) ** 2
    ).sum(axis=0) / np.maximum(nk, 1e-12)
    variances = np.maximum(variances, 1e-6)
    return weights, means, variances


def order_parameters(weights, means, variances):
    order = np.argsort(means)
    return weights[order], means[order], variances[order]


def em_gmm_1d(
    x,
    init_weights,
    init_means,
    init_variances,
    max_iter=200,
    tol=1e-6,
    cutoff=-10.0,
):
    weights = np.array(init_weights, dtype=float)
    means = np.array(init_means, dtype=float)
    variances = np.maximum(np.array(init_variances, dtype=float), 1e-6)
    weights = weights / weights.sum()

    log_likelihood_history = []

    for _ in range(max_iter):
        responsibilities, log_likelihood = e_step(
            x,
            weights,
            means,
            variances,
            cutoff=cutoff,
        )
        log_likelihood_history.append(log_likelihood)
        weights, means, variances = m_step(x, responsibilities)

        if len(log_likelihood_history) > 1:
            improvement = log_likelihood_history[-1] - log_likelihood_history[-2]
            if abs(improvement) < tol:
                break

    weights, means, variances = order_parameters(weights, means, variances)
    return {
        "weights": weights,
        "means": means,
        "variances": variances,
        "log_likelihood_history": np.array(log_likelihood_history),
        "iterations": len(log_likelihood_history),
    }


def print_parameter_comparison(title, result):
    print(title)
    print("true weights:     ", TRUE_WEIGHTS)
    print("estimated weights:", result["weights"])
    print("true means:       ", TRUE_MEANS)
    print("estimated means:  ", result["means"])
    print("true variances:   ", TRUE_VARIANCES)
    print("estimated vars:   ", result["variances"])
    print("final log-likelihood:", result["log_likelihood_history"][-1])
    print("iterations:", result["iterations"])


def plot_log_likelihood(histories, labels, title, filename):
    if plt is None or matplotlib is None:
        print("matplotlib is not installed, so the log-likelihood plot was skipped.")
        return

    plt.figure(figsize=(8, 5))
    for history, label in zip(histories, labels):
        plt.plot(history, linewidth=2, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        output_path = os.path.join(os.path.dirname(__file__), filename)
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def random_initialization(x, n_components=3, seed=None):
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_components))
    means = rng.choice(x, size=n_components, replace=False)
    global_var = np.var(x) + 1e-3
    variances = rng.uniform(0.5 * global_var, 1.5 * global_var, size=n_components)
    return order_parameters(weights, means, variances)


def run_em_experiment(x, init_weights, init_means, init_variances, label):
    result = em_gmm_1d(x, init_weights, init_means, init_variances)
    print_parameter_comparison(label, result)
    return result


def main():
    x20, components20 = sample_gmm(
        20,
        TRUE_WEIGHTS,
        TRUE_MEANS,
        TRUE_VARIANCES,
        seed=7,
    )
    print("20 generated samples:")
    print(x20)
    print("sample component counts:", np.bincount(components20, minlength=3))

    result20 = run_em_experiment(
        x20,
        TRUE_WEIGHTS,
        TRUE_MEANS,
        TRUE_VARIANCES,
        label="EM with 20 samples, initialized at the true parameters",
    )

    x200, components200 = sample_gmm(
        200,
        TRUE_WEIGHTS,
        TRUE_MEANS,
        TRUE_VARIANCES,
        seed=21,
    )
    result200 = run_em_experiment(
        x200,
        TRUE_WEIGHTS,
        TRUE_MEANS,
        TRUE_VARIANCES,
        label="EM with 200 samples, initialized at the true parameters",
    )

    plot_log_likelihood(
        [result20["log_likelihood_history"], result200["log_likelihood_history"]],
        ["20 samples", "200 samples"],
        "EM log-likelihood with true initialization",
        "hw_1_q4_true_init.png",
    )

    mean_error_20 = np.mean(np.abs(result20["means"] - TRUE_MEANS))
    mean_error_200 = np.mean(np.abs(result200["means"] - TRUE_MEANS))
    weight_error_20 = np.mean(np.abs(result20["weights"] - TRUE_WEIGHTS))
    weight_error_200 = np.mean(np.abs(result200["weights"] - TRUE_WEIGHTS))

    print("\nAnswer for part (b):")
    if mean_error_200 < mean_error_20 and weight_error_200 < weight_error_20:
        print(
            "With 200 samples, the parameter estimates are closer to the true mixture than with 20 samples."
        )
    else:
        print(
            "With 200 samples, the fit is generally more stable, though finite-sample noise can still affect some parameters."
        )
    print(
        "The larger sample gives EM more information, so the estimates are typically less noisy and more reliable."
    )

    restart_results = []
    for restart in range(10):
        init_w, init_m, init_v = random_initialization(
            x200,
            n_components=3,
            seed=100 + restart,
        )
        result = em_gmm_1d(x200, init_w, init_m, init_v)
        restart_results.append(result)
        print(
            f"restart {restart + 1:02d} final log-likelihood = "
            f"{result['log_likelihood_history'][-1]:.6f}"
        )
        print("  weights   =", result["weights"])
        print("  means     =", result["means"])
        print("  variances =", result["variances"])

    best_result = max(
        restart_results,
        key=lambda current: current["log_likelihood_history"][-1],
    )
    likelihoods = np.array(
        [current["log_likelihood_history"][-1] for current in restart_results]
    )

    plot_log_likelihood(
        [current["log_likelihood_history"] for current in restart_results],
        [f"restart {i + 1}" for i in range(len(restart_results))],
        "EM log-likelihood with 10 random restarts (200 samples)",
        "hw_1_q4_random_restarts.png",
    )

    print("\nBest restart summary:")
    print_parameter_comparison("Best of 10 random restarts", best_result)

    print("Answer for part (c):")
    if np.allclose(likelihoods, likelihoods[0], atol=1e-4, rtol=1e-6):
        print(
            "All 10 restarts converged to essentially the same likelihood and solution on this run."
        )
    else:
        print(
            "The 10 random restarts did not all converge to the same likelihood or the same parameters."
        )
    print(
        "This shows EM is sensitive to initialization and behaves like a local optimization method, not a guaranteed global optimizer."
    )


if __name__ == "__main__":
    main()
