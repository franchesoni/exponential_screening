import matplotlib.pyplot as plt
import numpy as np


class ESEstimator:
    def __init__(self, sigma=1, estimate_sigma_alpha=None, T0=3000, T=7000):
        # to use sigma set sigma=real_sigma and estimate_sigma=False
        # to estimate sigma use sigma=sigma_upper_bound
        self.sigma, self.estimate_sigma_flag = (sigma, estimate_sigma_alpha is not None)
        self.alpha = estimate_sigma_alpha
        self.T, self.T0 = T, T0

    def estimate_sigma(self, X, Y, method='exp', maxiter=10):
        n, M = X.shape
        assert method in ['exp', 'binary']
        self.estimate_sigma_flag = False  # we are doing it
        if method=='exp':  # exponential search
            was_up, first_iter = True, True  # assume we start up
            while True:
                es_estimate = self.estimate_from_data(
                    X, Y
                )  # compute an estimate using self.sigma
                sq_norm = np.sum((Y - X @ es_estimate) ** 2)
                denominator = n - np.sum(es_estimate > 1 / n)
                empirical_s2 = sq_norm / denominator

                if (
                    self.sigma ** 2 - empirical_s2 > self.alpha
                ):  # check if self.sigma**2 is a different enough upper bound
                    if not was_up:  # if we went up
                        return self.sigma
                    self.sigma = self.sigma / 2
                    was_up = True
                else:
                    if was_up:
                        if first_iter:
                            first_iter = False
                            self.sigma = self.sigma * 100  # increase a lot
                        else:
                            return self.sigma * 2
                    else:
                        self.sigma = self.sigma * 2  # increase
        elif method=='binary':
            cinf = 0  # current inferior bound
            while True:  # exponential ramp up
                es_estimate = self.estimate_from_data(X, Y)
                sq_norm = np.sum((Y - X @ es_estimate) ** 2)
                denominator = n - np.sum(es_estimate > 1 / n)
                empirical_s2 = sq_norm / denominator
                if self.sigma**2 - empirical_s2 > self.alpha:
                    break
                else:
                    cinf = self.sigma
                    self.sigma = self.sigma*2
            csup = self.sigma  # current superior bound
            # binary search [cinf, csup]
            for i in range(maxiter):
                candidate = (csup + cinf) / 2
                self.sigma = candidate
                es_estimate = self.estimate_from_data(X, Y)
                sq_norm = np.sum((Y - X @ es_estimate) ** 2)
                denominator = n - np.sum(es_estimate > 1 / n)
                empirical_s2 = sq_norm / denominator
                if self.sigma**2 - empirical_s2 > self.alpha:
                    csup = candidate
                else:
                    cinf = candidate
            return csup
 


    def estimate_from_data(self, X, Y, save_traj=None):
        n, M = X.shape
        if self.estimate_sigma_flag:
            self.sigma = self.estimate_sigma(X, Y)
        if save_traj:
            traj1 = []
        p = [np.zeros((M, 1))]  # initialize list of {p_t}_t
        estimators = []
        for t in range(self.T):
            # create Q_t
            Q_t = p[t].copy()  # Q starts from p_t
            randind = np.random.randint(0, len(Q_t))  # select random perturbation index
            Q_t[randind] = (Q_t[randind] + 1) % 2  # and moves across one edge
            # fit estimators
            theta_p_t, residual_p_t = self.fit_least_squares_estimator(X, Y, p[t])
            theta_Q_t, residual_Q_t = self.fit_least_squares_estimator(X, Y, Q_t)
            if residual_Q_t.size * residual_p_t.size == 0:
                residual_p_t = np.sum((Y - X @ theta_p_t)**2)
                residual_Q_t = np.sum((Y - X @ theta_Q_t)**2)
            # choose stochastically
            r = self.r_fn(p[t], residual_p_t, Q_t, residual_Q_t, M)
            if np.random.rand(1) < r:
                p.append(Q_t)
                estimators.append(theta_Q_t)
            else:
                p.append(p[t])
                estimators.append(theta_p_t)
            if save_traj:
                traj1.append(theta_p_t)
                if t == self.T - 1:
                    return np.mean(estimators[self.T0 :], axis=0), np.array(traj1), np.array(p)
        return np.mean(estimators[self.T0 :], axis=0)

    def r_fn(self, p_t, residual_p_t, Q_t, residual_Q_t, M):
        # # compute r(p, q)
        exppart = np.exp(
            1 / (4 * self.sigma ** 2) * (residual_p_t - residual_Q_t)
            + 0.5 * (p_t.sum() - Q_t.sum())
        )
        if p_t.sum() > 0:
            prod1part = (Q_t.sum() / p_t.sum()) ** Q_t.sum()
            prod2part = (p_t.sum() / (2 * np.e * M)) ** (Q_t.sum() - p_t.sum())
            piqpratio = prod1part * prod2part
        else:
            piqpratio = (Q_t.sum() / (2 * np.e * M)) ** (Q_t.sum())  # simplification
        v_Q_t_over_v_p = exppart * piqpratio
        print
        return min(v_Q_t_over_v_p, 1)

    def fit_least_squares_estimator(self, X, Y, p):
        """ We have the equation Y = X theta + noise. We want to use the theta_p
        which minimizes the least squared error. This is equivalent to finding
        theta in R^p which minimizes the least squared error when using X_p.
        """
        assert X.ndim == 2  # X shape = (n, M)
        assert Y.shape == (X.shape[0], 1)  # Y shape = (n, 1)
        assert p.shape == (X.shape[1], 1)  # p shape = (M, 1)
        X_p = X[:, p.flatten().astype(bool)]  # select columns by p
        theta_in_Rp, residual = np.linalg.lstsq(X_p, Y, rcond=None)[:2]
        theta_p = np.zeros((X.shape[1], 1))
        theta_p[p.astype(bool)] = theta_in_Rp.flatten()
        return theta_p, residual


# n, M, S, seed, which one works
# 20, 15, ,42  both
# 15, 15, ,42  both (and they are very similar estimations)
# 14, 15, ,42  oracle works perfect, ES works doubtfully
# 13, 15, ,42  oracle, ES worse than previous try
# 10, 15, ,42  oracle

# 200, 500, 20, with synthetic, it worked!

np.random.seed(42)
from testbed import SyntheticProblem
if __name__ == "__main__":
    # create toy test problem
    n, M, S = 100, 200, 10  # n, M, S = 200, 500, 20
    sp = SyntheticProblem(n, M, S)
    X, Y = sp.get_data(n_exps=1)
    sigma = np.sqrt(S / 9)
    X, Y = X[0], Y[0]
    es = ESEstimator(sigma=sigma*10, estimate_sigma_alpha=1)
    # test estimate from data
    theta_es = es.estimate_from_data(X, Y)
    theta_star = np.concatenate((np.ones(S), np.zeros(M-S)), axis=0).reshape(-1, 1)

    # print results
    print('true', theta_star)
    print('estimated known variance', theta_es)

    plt.figure()
    plt.bar(np.arange(M), theta_star.flatten(), width=1, color=(1, 0, 0, 0.2), label='true')
    plt.bar(np.arange(M), theta_es.flatten(), width=0.5, color=(0, 0, 1, 0.2), label='ES')
    plt.legend()
    plt.show()
