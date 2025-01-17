import numpy as np
import sympy as sp
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from Procedures.ldl_decomposition import ldl_decomposition, ldl_solve


class LevenbergMarquardtSolver:
    def __init__(self, f, variables, x0,
                 epsilon=0.5e-5, delta=0.5e-5, eta=0.5e-5,
                 maxiter=100, use_symbolic=True):
        """
        Initialize the Levenberg-Marquardt solver.

        Parameters:
        ----------
        f : sympy.Expr
            The scalar function to minimize.
        variables : list of sympy.Symbol
            The variables of the function.
        x0 : array_like
            Initial guess for the minimizer.
        epsilon : float, optional
            Tolerance for step size (||x_{k+1} - x_k||).
        delta : float, optional
            Tolerance for function value change (|f(x_{k+1}) - f(x_k)|).
        eta : float, optional
            Tolerance for gradient norm (||∇f(x_k)||).
        maxiter : int, optional
            Maximum number of iterations.
        """
        self.f = f
        self.variables = variables
        self.x = np.array(x0, dtype=float)
        self.epsilon = epsilon
        self.delta = delta
        self.eta = eta
        self.maxiter = maxiter
        self.use_symbolic = use_symbolic
        self.gradient = None
        self.hessian = None
        self.history = []  # To store intermediate results

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

    def line_search(self, f_func, x, s):
        """
        Finds the minimum of the function f, starting from x, in the direction s.

        Arguments:
            f_func - Lambdified function to minimize
            x      - NumPy array (current position)
            s      - NumPy array (search direction)

        Returns:
            Minimiser (NumPy array) and whether the search was successful
        """
        def g(t):
            return f_func(*(x + t * s))  # Unpack x + t * s into f_func

        # Perform the line search
        result = minimize_scalar(g)

        # Update x using the minimiser t
        t_min = result.x
        return x + t_min * s, result.success

    def solve(self, verbose=False):
        """Run the Levenberg-Marquardt method."""
        f_func = sp.lambdify(self.variables, self.f, 'numpy')
        f_current = f_func(*self.x)

        for iteration in range(self.maxiter):
            # Compute gradient and Hessian
            grad_eval = np.array([g(*self.x) for g in self.grad_func])
            H_eval = np.array(self.hessian_func(*self.x))

            # Perform LDL decomposition
            try:
                L, D, p = ldl_decomposition(H_eval, verbose=verbose)
            except Exception as e:
                return self._finalize(iteration, f_current, f"LDL decomposition failed: {e}")

            # If not positive definite, update H to be H + a I
            if np.any(np.diag(D) <= 0):
                a = 1.1 * norm(H_eval, np.inf)
                H_eval += a * np.identity(H_eval.shape[0])
                L, D, p = ldl_decomposition(H_eval)

            # Solve H_eval * d = -grad_eval using LDL^T
            d = ldl_solve(L, D, p, -grad_eval)

            # Perform line search in the direction d
            x_new, success = self.line_search(f_func, self.x, d)
            if not success:
                return self._finalize(iteration, f_current, "Line search failed")

            # Evaluate the function at the new point
            f_new = f_func(*x_new)

            # Store history
            self.history.append((self.x.copy(), f_current))

            if verbose:
                print(f"Iteration {iteration + 1}: x = {x_new}, f(x) = {f_new}")

            # Check convergence
            grad_eval_new = np.array([g(*x_new) for g in self.grad_func])
            if np.linalg.norm(grad_eval_new, 2) < self.eta and \
                    np.linalg.norm(x_new - self.x, 2) < self.epsilon and \
                    abs(f_new - f_current) < self.delta:
                self.x = x_new  # Ensure self.x is updated to the final value
                return self._finalize(iteration + 1, f_new, "Converged within tolerances")

            # Accept new point
            self.x = x_new
            f_current = f_new

        return self._finalize(self.maxiter, f_current, "Maximum iterations reached")

    def _finalize(self, iterations, f_min, reason):
        """
        Finalize and return the result.

        Returns
        -------
        result : dict
            A dictionary containing:
            - 'x_min': np.ndarray, the estimated minimizer.
            - 'iterations': int, the number of iterations taken.
            - 'success': bool, whether the method converged successfully.
            - 'f_min': float, the function value at the final point.
            - 'reason': str, a message describing why the method terminated.
        """
        success_reasons = [
            "Converged within tolerances",
        ]
        return {
            "x_min": self.x,
            "iterations": iterations,
            "success": reason in success_reasons,
            "f_min": f_min,
            "reason": reason,
            "history": self.history
        }