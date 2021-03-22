import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.model_selection


class DigitsClassificationProblem:
    def __init__(self):
        pass

    def get_data(self, test_qty=1e100, normalize=False):
        X0, labels = sklearn.datasets.load_digits(n_class=10, return_X_y=True)
        if normalize:
            X0 = X.copy() / 16
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X0, labels, test_size=0.5, shuffle=False)
        X, labels_train, Y, labels_test = X_train.T, y_train, X_test[:test_qty], y_test[:test_qty]
        return (
            X, Y, labels_train, labels_test
        )





class DigitsDenoisingProblem:
    def __init__(self, sigma):
        self.sigma = sigma

    def get_data(self, n_exps=1, obs_index=None, normalize=True):
        X, labels = sklearn.datasets.load_digits(n_class=10, return_X_y=True)
        if normalize:
            X = X.copy() / 16
        xi = np.random.randn(n_exps, X.shape[-1], 1)
        if obs_index is None:
            obs_index = np.random.randint(0, len(labels))
        true_digit, label = X[obs_index], labels[obs_index]
        X_without_true_digit, labels_without_true_digit = (
            np.concatenate((X[:obs_index], X[obs_index + 1 :]), axis=0),
            np.concatenate((labels[:obs_index], labels[obs_index + 1 :]), axis=0),
        )
        Y = true_digit.reshape(-1, 1) + self.sigma * xi  # this broadcasts
        return (
            X_without_true_digit.reshape(-1, X.shape[-1], 1).T,
            Y,
            true_digit,
            label,
            labels_without_true_digit,
        )
        # output shape is (1, 64, 1000+)  (n_exps, 64, 1000+)

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


class RandintProblem:
    def __init__(self, sigma, n, M, S, maxint):
        self.sigma = sigma
        self.n = n
        self.M = M
        self.S = S
        self.maxint = maxint

    def get_data(self, n_exps=1):
        theta_star = np.random.randint(
            1, self.maxint + 1, (n_exps, self.M, 1)
        )  # genreate random ints
        p_star = np.concatenate(
            (np.ones(self.S), np.zeros(self.M - self.S)), axis=0
        ).reshape(-1, 1)
        theta_star[:, ~(p_star.astype(bool))] = 0
        X = np.random.randn(n_exps, self.n, self.M)
        Y = X @ theta_star + self.sigma * np.random.randn(n_exps, self.n, 1)
        return X, Y, theta_star, p_star


if __name__ == "__main__":  # test things
    rp = RandintProblem(1, 10, 20, 2, 9)
    X, Y, theta_star, p_star = rp.get_data(n_exps=2)

    sp = SyntheticProblem(100, 200, 10)
    X, Y = sp.get_data()

    dsigma = 1
    dp = DigitsDenoisingProblem(sigma=dsigma)
    X, Y, true_digit, label = dp.get_data(n_exps=3)
    dp.visualize_digits(Y, true_digit, label)

