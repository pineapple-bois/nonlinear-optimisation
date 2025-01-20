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

    def solve(self, verbose=False):
        """
        Run a trust-region-style Levenberg-Marquardt method.
        We'll adapt lambda up/down based on actual vs. predicted reduction.
        """
        f_func = sp.lambdify(self.variables, self.f, 'numpy')
        f_current = self._evaluate_function(f_func, self.x)
        # If self.lamb is None or 0, pick some default
        if not self.lamb:
            self.lamb = 1.1

        for iteration in range(self.maxiter):
            # 1. Evaluate gradient/Hessian
            grad_eval = np.array([g(*self.x) for g in self.grad_func])
            H_eval = np.array(self.hessian_func(*self.x))

            # 2. Attempt to find a positive-definite H_aug
            for _ in range(5):  # up to 5 tries
                a = self.lamb * norm(H_eval, np.inf)
                H_aug = H_eval + a * np.eye(len(self.x))
                try:
                    L, D, p = ldl_decomposition(H_aug, verbose=verbose)
                    # success, break the loop
                    break
                except:
                    # increase lamb, try again
                    self.lamb *= 2
            else:
                # If we never broke out of the loop => fail
                return self._finalize(iteration, f_current,
                                      "Indefinite H after repeated shifting")

            # 3. Solve for d using LDL decomposition
            try:
                d = ldl_solve(L, D, p, -grad_eval)
            except Exception as e:
                return self._finalize(
                    iteration, f_current,
                    f"LDL solve failed: {e}"
                )

            # 4. Predicted reduction:
            pred_reduction = -grad_eval.dot(d) - 0.5 * d.dot(H_aug.dot(d))

            # 5. Evaluate at x_trial
            x_trial = self.x + d
            f_trial = self._evaluate_function(f_func, x_trial)
            actual_reduction = f_current - f_trial

            rho = 0 if (pred_reduction == 0) else actual_reduction / pred_reduction

            if verbose:
                print(f"Iter {iteration}: f={f_current:.5g}, "
                      f"pred={pred_reduction:.3g}, actual={actual_reduction:.3g}, "
                      f"rho={rho:.2g}, lambda={self.lamb:.3g}")

            # 6. Accept or reject
            if rho < 0.25:
                # reject step, increase lamb
                self.lamb *= 2
            else:
                # accept step
                self.x = x_trial
                f_current = f_trial

                # update lamb if step is good
                if rho > 0.75:
                    self.lamb *= 0.5

                # store history if requested
                if self.track_history:
                    self._store_history(f_current, grad_eval, iteration)

                # 7. Check convergence
                grad_eval_new = np.array([g(*self.x) for g in self.grad_func])
                if (np.linalg.norm(grad_eval_new, 2) < self.eta
                        and np.linalg.norm(d, 2) < self.epsilon
                        and abs(actual_reduction) < self.delta):
                    return self._finalize(iteration + 1, f_current,
                                          "Converged within tolerances")
        # If we exit the for-loop, finalize:
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
