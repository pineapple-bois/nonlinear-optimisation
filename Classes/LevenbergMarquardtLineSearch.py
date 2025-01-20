import numpy as np
import sympy as sp
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from Procedures.ldl_decomposition import ldl_decomposition, ldl_solve


class LevenbergMarquardtSolver:
    def __init__(self, f, variables, x0, lamb=1.1,
                 epsilon=0.5e-5, delta=0.5e-5, eta=0.5e-5,
                 maxiter=100, use_symbolic=True, track_history=True):
        """
        Initialise the Levenberg-Marquardt solver.

        Parameters:
        ----------
        f : sympy.Expr
            The scalar function to minimise.
        variables : list of sympy.Symbol
            The variables of the function.
        x0 : array_like
            Initial guess for the minimiser.
        lamb: float
            Damping parameter of augmented Hessian
        epsilon : float, optional
            Tolerance for step size (||x_{k+1} - x_k||).
        delta : float, optional
            Tolerance for function value change (|f(x_{k+1}) - f(x_k)|).
        eta : float, optional
            Tolerance for gradient norm (||∇f(x_k)||).
        maxiter : int, optional
            Maximum number of iterations.
        track_history : bool, optional
            Whether to store history of iterations for analysis.
        """
        self.f = f
        self.variables = variables
        self.x = np.array(x0, dtype=float)
        self.epsilon = epsilon
        self.delta = delta
        self.eta = eta
        self.maxiter = maxiter
        self.use_symbolic = use_symbolic
        self.track_history = track_history
        self.nfev = 0
        self.gradient = None
        self.hessian = None
        self.history = []  # To store intermediate results
        self.lamb = lamb

        if self.use_symbolic:
            self._compute_symbolic_derivatives()
        else:
            self.grad_func = None
            self.hessian_func = None

    def _compute_symbolic_derivatives(self):
        """Compute symbolic gradient and Hessian."""
        self.gradient = sp.Matrix([sp.diff(self.f, var) for var in self.variables])
        self.hessian = sp.Matrix([
            [sp.diff(self.f, var1, var2) for var2 in self.variables]
            for var1 in self.variables
        ])
        self.grad_func = [sp.lambdify(self.variables, g, 'numpy') for g in self.gradient]
        self.hessian_func = sp.lambdify(self.variables, self.hessian, 'numpy')

    def _evaluate_function(self, f_func, x):
        """Evaluate the function and increment the counter."""
        self.nfev += 1
        return f_func(*x)

    def _backtracking_search(self, f_func, x, grad, d, f0=None, c=1e-4, tau=0.5, max_ls=10):
        """
        A simple backtracking line search enforcing the Armijo condition:
            f(x + alpha*d) <= f(x) + c * alpha * grad(x)^T d

        Parameters
        ----------
        f_func : callable
            A function that takes 'x' and returns the scalar objective f(x).
        x : np.ndarray
            Current point.
        grad : np.ndarray
            Gradient at the current point x.
        d : np.ndarray
            Search direction.
        f0 : float, optional
            The current function value f(x). If not given, we compute f(x).
        c : float, optional
            Armijo constant (typ. 1e-4).
        tau : float, optional
            Reduction factor (typ. 0.5).
        max_ls : int, optional
            Maximum number of backtracking reductions.

        Returns
        -------
        x_new : np.ndarray
            The new point after the line search.
        success : bool
            Whether we found a satisfactory alpha within max_ls tries.
        alpha : float
            The final step length used.
        """
        # If we don't have f0, compute it once
        if f0 is None:
            f0 = self._evaluate_function(f_func, x)

        # Compute directional derivative g^T d
        dir_deriv = grad.dot(d)

        alpha = 1.0
        for _ in range(max_ls):
            x_trial = x + alpha * d
            f_trial = self._evaluate_function(f_func, x_trial)

            # Armijo condition
            if f_trial <= f0 + c * alpha * dir_deriv:
                # success
                return x_trial, True, alpha

            alpha *= tau  # reduce step and try again

        # if we exit loop => failed to satisfy condition
        return x + alpha * d, False, alpha

    def solve(self, verbose=False):
        """
        Run a 'hybrid' approach:
          - Use a Levenberg–Marquardt style damping: (H + lambda*I)*d = -grad
          - Then do a line search along d
          - If the new point doesn't improve the function, increase lambda & retry
          - If it improves, accept & possibly decrease lambda
        """

        f_func = sp.lambdify(self.variables, self.f, 'numpy')
        f_current = self._evaluate_function(f_func, self.x)

        if not self.lamb:
            self.lamb = 1.0

        for iteration in range(self.maxiter):
            # compute gradient / Hessian at current x
            grad_eval = np.array([g(*self.x) for g in self.grad_func])
            H_eval = np.array(self.hessian_func(*self.x))

            # check convergence by gradient norm
            if norm(grad_eval, 2) < self.eta:
                return self._finalize(iteration, f_current, "Converged within tolerances")

            # We may need an inner loop to handle indefinite or unsuccessful steps
            attempt = 0
            max_attempts = 5  # or however many times you want to retry
            while attempt < max_attempts:
                # 1) Form (H + lambda * I)
                a = self.lamb * norm(H_eval, np.inf)
                H_aug = H_eval + a * np.eye(len(self.x))

                # 2) Factor H_aug and solve for d
                try:
                    L, D, p = ldl_decomposition(H_aug, verbose=verbose)
                    d = ldl_solve(L, D, p, -grad_eval)
                except Exception as e:
                    # If factorization fails, maybe we just increase lambda and retry
                    if verbose:
                        print(f"LDL decomposition failed with error {e}, doubling lambda and retry.")
                    self.lamb *= 2
                    attempt += 1
                    continue

                # 3) Backtracking line search
                x_new, success, alpha = self._backtracking_search(f_func, self.x, grad_eval, d, f0=f_current)

                if not success:
                    # line search didn't find a good step => treat as "no improvement"
                    if verbose:
                        print("No acceptable step found via backtracking; increasing lambda.")
                    self.lamb *= 2
                    attempt += 1
                    continue

                # evaluate function at new point
                f_new = self._evaluate_function(f_func, x_new)

                # 4) If improvement => accept & reduce lambda
                if f_new < f_current:
                    # Accept
                    if verbose:
                        print(f"Iteration {iteration}: improved from {f_current:.4g} to {f_new:.4g} "
                              f"with lambda={self.lamb:.3g}")
                    self.x = x_new
                    self.lamb *= 0.8  # mild decrease
                    f_current = f_new
                    break  # break out of inner loop, proceed to next iteration
                else:
                    # No improvement => reject & increase lambda
                    if verbose:
                        print(f"No improvement, new f={f_new:.4g} >= old f={f_current:.4g}, "
                              f"increasing lambda to {self.lamb:.3g} and retrying.")
                    self.lamb *= 2
                    attempt += 1

            else:
                # We exhausted max_attempts without improvement
                return self._finalize(iteration, f_current,
                                      "Failed to improve after multiple attempts.")

            # after we accept the step, store history if desired
            if self.track_history:
                self._store_history(f_current, grad_eval, iteration)

            # check for convergence on function, step size, gradient, etc.
            grad_eval_new = np.array([g(*self.x) for g in self.grad_func])
            if (norm(grad_eval_new, 2) < self.eta and
                    norm(d, 2) < self.epsilon and
                    abs(f_new - f_current) < self.delta):
                return self._finalize(iteration + 1, f_new, "Converged within tolerances")

        # If loop finishes with no return => max iters
        return self._finalize(self.maxiter, f_current, "Maximum iterations reached")

    def _store_history(self, f_current, grad_eval, iteration):
        """Store the current state for debugging and analysis."""
        self.history.append({
            "iteration": iteration,
            "x": self.x.copy(),
            "f": f_current,
            "grad": grad_eval.copy(),
            "lambda": self.lamb
        })

    def _finalize(self, iterations, f_min, reason):
        """
        Finalize and return the result.

        Returns
        -------
        result : dict
            A dictionary containing:
            - 'message': str, a message describing why the method terminated.
            - 'success': bool, whether the method converged successfully.
            - 'x': np.ndarray, the estimated minimizer.
            - 'fun': float, the function value at the final point.
            - 'nit': int, number of iterations taken.
            - 'nfev': int, number of function evaluations.
            - 'lambda': float, final value of lambda.
            - 'history': list, intermediate results for analysis.
        """
        success_reasons = [
            "Converged within tolerances",
        ]
        return {
            "success": reason in success_reasons,
            "reason": reason,
            "x": self.x,
            "fun": f_min,
            "nit": iterations,
            "nfev": self.nfev,
            "lambda": self.lamb,
            "history": self.history
        }
