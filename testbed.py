import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


class DigitsProblem:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_data(self, n_exps=1, obs_index=None):
        X, labels = sklearn.datasets.load_digits(n_class=10, return_X_y=True)
        xi = np.random.randn(n_exps, X.shape[-1], 1)
        if obs_index is None:
            obs_index = np.random.randint(0, len(labels))
        true_digit, label = X[obs_index], labels[obs_index]
        X_without_true_digit = np.concatenate(
            (X[:obs_index], X[obs_index + 1 :]), axis=0
        )
        Y = true_digit.reshape(-1, 1) + self.sigma * xi  # this broadcasts
        return X_without_true_digit.reshape(-1, X.shape[-1], 1), Y, true_digit, label

    def visualize_digits(self, Y, true_digit, label):
        imgs = Y.reshape(-1, 8, 8)
        for img in imgs:
            plt.figure(figsize=(4, 2))
            plt.subplot(1, 2, 1)
            plt.title(f"$\sigma={self.sigma}$")
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.title(f"labeled as {label}")
            plt.imshow(true_digit.reshape(8, 8), cmap="gray")
            plt.axis("off")
            plt.show()




class SyntheticProblem:
    def __init__(self, n, M, S, X_type="gaussian"):
        """Initialization

        Args:
            n (int): output dimensionality
            M (int): number of covariates
            S (int): number of non-zero entries in target coefficient vector
            X_type (str, optional): type of independent variables X cointains.
                Defaults to 'gaussian'.
        """
        assert X_type in ["rademacher", "gaussian"]
        self.X_type = X_type
        self.n, self.M, self.S = n, M, S
        self.sigma = np.sqrt(S / 9)
        self.theta_star = np.concatenate(
            (np.ones(self.S), np.zeros(self.M - self.S)), axis=-1
        ).reshape(M, 1)

    def get_data(self, n_exps=1):
        """ Returns data as in Sparse Recovery (section 7.2.1).

        Args:
            n_exps (int, optional): number of experiments to be returned,
                aka batch size. Defaults to 500.

        Returns:
            (np.ndarray): X. Random matrix of shape (n_exps, n, M) cointaining 
                independent Gaussian or Rademacher variables.
            (np.ndarray): Y. Matrix of observations. Y X theta + sigma xi
        """
        xi = np.random.randn(n_exps, self.n, 1)
        if self.X_type == "gaussian":
            X = np.random.randn(n_exps, self.n, self.M)
        elif self.X_type == "rademacher":
            X = np.random.randint(0, 2, (n_exps, self.n, self.M)) * 2 - 1
        Y = X @ self.theta_star + self.sigma * xi
        return X, Y



if __name__ == "__main__":
    sp = SyntheticProblem(100, 200, 10)
    X, Y = sp.get_data()

    dsigma = 1
    dp = DigitsProblem(sigma=dsigma)
    X, Y, true_digit, label = dp.get_data(n_exps=3)
    dp.visualize_digits(Y, true_digit, label)

