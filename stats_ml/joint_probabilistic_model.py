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


def sigmoid(t):
    t = np.clip(t, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-t))


def generate_data(n_samples, w_true, b_true, mu_true, seed=42):
    rng = np.random.default_rng(seed)

    x = rng.normal(loc=0.0, scale=1.5, size=n_samples)
    logits = w_true * x + b_true
    probs = sigmoid(logits)
    y = rng.binomial(n=1, p=probs)

    z = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    positive_mask = y == 1
    z[positive_mask] = rng.normal(
        loc=mu_true, scale=1.0, size=positive_mask.sum()
    )

    return x, y, z


def estimate_mu(y, z):
    positive_count = y.sum()
    if positive_count == 0:
        raise ValueError("Cannot estimate mu because there are no positive samples.")
    return np.sum(y * z) / positive_count


def logistic_log_likelihood(x, y, w, b):
    logits = w * x + b
    return np.sum(y * logits - np.logaddexp(0.0, logits))


def train_logistic_regression(x, y, learning_rate=0.01, epochs=2000):
    w = 0.0
    b = 0.0
    log_likelihood_history = []

    n_samples = x.shape[0]

    for epoch in range(epochs):
        sigma = sigmoid(w * x + b)
        grad_w = np.sum(x * (y - sigma))
        grad_b = np.sum(y - sigma)

        w += learning_rate * grad_w / n_samples
        b += learning_rate * grad_b / n_samples

        ll = logistic_log_likelihood(x, y, w, b)
        log_likelihood_history.append(ll)

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | log-likelihood = {ll:.4f} | "
                f"w = {w:.4f}, b = {b:.4f}"
            )

    return w, b, np.array(log_likelihood_history)


def plot_training_curve(log_likelihood_history):
    if plt is None or matplotlib is None:
        print("matplotlib is not installed, so the training curve was not plotted.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(log_likelihood_history, color="tab:blue", linewidth=2)
    plt.title("Log-Likelihood During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Log-Likelihood")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        output_path = os.path.join(
            os.path.dirname(__file__), "joint_model_log_likelihood.png"
        )
        plt.savefig(output_path, dpi=150)
        print(f"Saved training curve to {output_path}")
    else:
        plt.show()


def main():
    n_samples = 1000
    w_true = 2.0
    b_true = -0.7
    mu_true = 2.5

    learning_rate = 0.1
    epochs = 3000

    x, y, z = generate_data(
        n_samples=n_samples,
        w_true=w_true,
        b_true=b_true,
        mu_true=mu_true,
        seed=42,
    )

    mu_est = estimate_mu(y, z)
    w_est, b_est, log_likelihood_history = train_logistic_regression(
        x=x,
        y=y,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    print("\nParameter comparison")
    print(f"w_true  = {w_true: .4f} | w_est  = {w_est: .4f}")
    print(f"b_true  = {b_true: .4f} | b_est  = {b_est: .4f}")
    print(f"mu_true = {mu_true: .4f} | mu_est = {mu_est: .4f}")

    plot_training_curve(log_likelihood_history)


if __name__ == "__main__":
    main()
