# Classes Defined:

## 1. Trust-Region-Style [Levenberg–Marquardt Method](LevenbergMarquardtTrustRegion.py)

This Python implementation of a **trust-region–style** Levenberg–Marquardt (LM) algorithm modifies the Hessian by adding a diagonal shift to ensure positive definiteness. It is well-suited for solving nonlinear least-squares or general unconstrained optimisation problems, especially in cases where the Hessian is indefinite or nearly singular.

The LM method combines **Newton’s method** and **gradient descent**. We solve

$$
(\mathbf{H} + \lambda \mathbf{I}) \mathbf{d} = -\nabla f
$$

where $\mathbf{H}$ is the Hessian and $\lambda$ is a *damping* parameter. Adjusting $\lambda$ allows the algorithm to move between small, “gradient-descent–like” steps (large $\lambda$) and near-Newton steps (small $\lambda$).

### Trust-Region Acceptance Criterion

The trust-region approach evaluates whether the proposed step $\mathbf{d}$ is acceptable by comparing:
- The **actual reduction** in function value:

$$
\text{actual} = f(\mathbf{x}) - f(\mathbf{x} + \mathbf{d})
$$

- The **predicted reduction** from the quadratic model:

$$
\text{predicted} = \nabla f(\mathbf{x})^T \mathbf{d} - \frac{1}{2} \mathbf{d}^T \mathbf{H}_{\text{aug}} \mathbf{d}
$$

The ratio

$$
\rho = \frac{\text{actual}}{\text{predicted}}
$$

determines how the damping parameter $\lambda$ is adjusted:
- $\rho > 0.75$: The step is acceptable and $\lambda$ is decreased.
- $\rho < 0.25$: The step is rejected, $\mathbf{x}$ remains unchanged, and $\lambda$ is increased.
- $0.25 \leq \rho \leq 0.75$: The step is acceptable, but $\lambda$ remains unchanged.

---

### Key Components

1. **Symbolic Derivatives**  
   - Gradients and Hessians are computed symbolically using [SymPy](https://www.sympy.org/), ensuring precision for small-scale problems.

2. **$\mathbf{LDL}^T$ Decomposition**  
   - The augmented Hessian $\mathbf{H}_{\text{aug}} = \mathbf{H} + \lambda \mathbf{I}$ is factored using a custom $\mathbf{LDL}^T$ decomposition.
   - Pivoting ensures positive definiteness when $\mathbf{H}$ is indefinite or near-singular.

3. **Adaptive Damping**  
   - The shift $a = \lambda \cdot \|\mathbf{H}\|_\infty$ ensures that the damping parameter $\lambda$ adapts to the scale of the Hessian.

4. **Acceptance or Rejection of Steps**  
   - Steps are evaluated based on $\rho$. Rejected steps increase $\lambda$ to reduce the step size, while successful steps decrease $\lambda$ to take larger steps.

5. **Convergence Checks**  
   - The algorithm terminates when:
     - Gradient norm: $\|\nabla f(\mathbf{x})\| < \eta$
     - Step size: $\|\mathbf{d}\| < \epsilon$
     - Function value change: $|f(\mathbf{x+d}) - f(\mathbf{x})| < \delta$

---

### Pseudocode

1. **Initialise**:
   - Set $\mathbf{x}_0$ (initial guess), $\lambda$ (damping parameter), and tolerances $\epsilon$, $\delta$, $\eta$.
   - Define the maximum number of iterations, `maxiter`.

2. **Iterate**:
   - Compute $\nabla f(\mathbf{x}_k)$ and $\mathbf{H}(\mathbf{x}_k)$.
   - Form $\mathbf{H}_{\text{aug}} = \mathbf{H} + a \mathbf{I}$, where $a = \lambda \|\mathbf{H}\|_\infty$
   - Factor $\mathbf{H}_{\text{aug}}$ using $\mathbf{LDL}^T$ decomposition.
   - Solve $\mathbf{H}_{\text{aug}} \mathbf{d} = -\nabla f(\mathbf{x}_k)$
   - Compute the predicted reduction.
   - Evaluate $f(\mathbf{x}_k + \mathbf{d})$ and compute the actual reduction.
   - Calculate $\rho = \frac{\text{actual}}{\text{predicted}}$.
   - Adjust $\lambda$:
     - If $\rho < 0.25$, reject the step and increase $\lambda$.
     - If $\rho > 0.75$, accept the step and decrease $\lambda$.
     - Otherwise, accept the step but leave $\lambda$ unchanged.
   - Check convergence criteria; if satisfied, terminate.

3. **Return**:
   - Best $\mathbf{x}$, final function value, iteration count, and history (if enabled).

---

### When to Use

- **Nonlinear Least Squares**: Ideal for residual-minimisation and curve-fitting tasks.
- **General Smooth Objectives**: Extends to any smooth function $f$, provided the Hessian can be computed.
- **Ill-Conditioned Hessians**: Effective for near-singular problems due to adaptive damping.

### Advantages

- **Robust**: Trust-region adjustments prevent overly large, unstable steps.
- **Efficient**: Balances between Newton-like steps (fast convergence near solutions) and gradient-descent-like steps for poor initial guesses.
- **Scale-Invariant**: Automatically adjusts $\lambda$ based on the magnitude of $\mathbf{H}$.

### Limitations

- Symbolic differentiation can be computationally expensive for high-dimensional problems.
- The trust-region approach may require multiple iterations to adaptively scale $\lambda$ in poorly scaled regions.

---

### Example Usage

```python
from lm_solver import LevenbergMarquardtSolver
import sympy as sp

# Define a simple function f(x, y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = LevenbergMarquardtSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],  # initial guess
    lamb=1.0,       # initial damping parameter
    maxiter=100,
    track_history=True
)

# Solve
result = solver.solve(verbose=True)
print("Solution:", result["x"])
print("Objective value:", result["fun"])
print("Number of iterations:", result["nit"])
```

----

## 2. Backtracking-Line-Search Style [Levenberg–Marquardt Method](LevenbergMarquardtLineSearch.py)

This Python implementation of a **hybrid** Levenberg–Marquardt (LM) method combines a damped Newton step with a **backtracking line search** to refine the step length. It handles indefiniteness in the Hessian (via damping) and avoids overly large steps by satisfying the **Armijo condition**.

### Method Overview

1. Compute a damped Newton direction by solving:

$$
(\mathbf{H} + \lambda \mathbf{I}) \, \mathbf{d} = -\nabla f,
$$

where $\mathbf{H}$ is the Hessian, $\lambda \mathbf{I}$ is the damping term, and $\mathbf{d}$ is the step direction.

2. Apply a backtracking line search to find a step length $\alpha$ such that:

$$
f(\mathbf{x} + \alpha \mathbf{d}) \leq f(\mathbf{x}) + c \alpha \nabla f(\mathbf{x})^T \mathbf{d}.
$$

3. If the line search fails:
   - Reject the step,
   - Increase $\lambda$, and
   - Retry the iteration with a smaller step size.

4. If the step succeeds:
   - Accept the step and move to the next iteration,
   - Optionally decrease $\lambda$ for less damping in subsequent steps.

---

### Key Components

1. **Symbolic Derivatives**  
   - Gradients $\nabla f$ and Hessians $\mathbf{H}$ are generated using [SymPy](https://www.sympy.org/), ensuring correctness for small-scale problems.

2. **$\mathbf{LDL}^T$ Decomposition**  
   - The augmented Hessian $\mathbf{H}_{\text{aug}} = \mathbf{H} + \lambda \mathbf{I}$ is factored via a custom $\mathbf{LDL}^T$ decomposition for numerical stability.

3. **Backtracking Line Search**  
   - Implements the **Armijo condition** to refine the step length $\alpha$:

$$
f(\mathbf{x} + \alpha \mathbf{d}) \leq f(\mathbf{x}) + c \alpha \nabla f(\mathbf{x})^T \mathbf{d}.
$$

   - If no suitable $\alpha$ is found, the step is rejected, $\lambda$ is increased, and the iteration is retried.

4. **Adaptive Damping**  
   - If a step is successful, $\lambda$ is decreased (e.g., $\lambda \leftarrow 0.8 \lambda$).  
   - If the step fails, $\lambda$ is doubled to ensure stability.

5. **Convergence Criteria**  
   - The algorithm stops when:
     - $\|\nabla f(\mathbf{x})\|$ is below a tolerance,
     - $\|\mathbf{d}\|$ is sufficiently small, or
     - The change in $f(\mathbf{x})$ is negligible.

---

### Pseudocode

1. **Initialise**:
   - $\mathbf{x}_0$ (initial guess), $\lambda$ (damping parameter), tolerances ($\varepsilon, \delta, \eta$), and maximum iterations (`maxiter`).

2. **Iteration** (for $k = 1, 2, \dots, \text{maxiter}$):
   - Compute $\nabla f(\mathbf{x}_k)$ and $\mathbf{H}(\mathbf{x}_k)$.
   - Form the augmented Hessian: 

$$
\mathbf{H}_{\text{aug}} = \mathbf{H} + \lambda \|\mathbf{H}\|_\infty \mathbf{I}.
$$

   - Factor $\mathbf{H}_{\text{aug}}$ using $\mathbf{LDL}^T$ decomposition.
   - Solve $\mathbf{H}_{\text{aug}} \mathbf{d} = -\nabla f(\mathbf{x}_k)$.
   - Perform a backtracking line search to find $\alpha$ satisfying the Armijo condition.
   - If the step fails:
     - Reject the step,
     - Increase $\lambda$, and retry.
   - If the step succeeds:
     - Accept the step: $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha \mathbf{d}$.
     - Optionally decrease $\lambda$.
   - Check convergence criteria.

3. **Return**:
   - $\mathbf{x}$ (solution), $f(\mathbf{x})$, number of iterations, and function evaluations.

---

### When to Use

- **Ill-Conditioned Problems**: Handles near-singular Hessians or indefinite cases with damping.
- **General Smooth Functions**: Works for any scalar objective $f$ where gradients and Hessians are available.
- **Avoiding Large Steps**: Line search ensures stability by regulating step lengths.

---

### Advantages

- **Combines Damping and Line Search**:
  - Damping stabilises the Newton step, while the line search ensures step lengths satisfy a sufficient decrease condition.
- **Adaptive**:
  - Dynamically adjusts $\lambda$ based on the success or failure of each step.
- **Symbolic Precision**:
  - Automatic differentiation reduces errors in calculating $\nabla f$ and $\mathbf{H}$.

---

### Caveats

- **Multiple Factorisations**:
  - If a step fails, the augmented Hessian must be re-factored with a larger $\lambda$.
- **Increased Function Evaluations**:
  - Line search requires multiple evaluations of $f(\mathbf{x} + \alpha \mathbf{d})$, which may be expensive for complex functions.
- **Parameter Sensitivity**:
  - Parameters like the backtracking constant ($c$) and damping adjustment factors may need tuning for optimal performance.

---

### Example Usage

```python
import sympy as sp
from LevenbergMarquardtLineSearch import LevenbergMarquardtSolver

# Define the function f(x, y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = LevenbergMarquardtSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],    # Initial guess
    lamb=1.0,         # Initial damping
    maxiter=100,
    track_history=True
)

# Solve
result = solver.solve(verbose=True)
print("Solution:", result["x"])
print("Objective value:", result["fun"])
print("Iterations:", result["nit"])
print("Function evaluations:", result["nfev"])
```

----

## 3. [Newton–Raphson Method](NewtonRaphson.py)

This Python implementation of the **Newton–Raphson method** iteratively refines a guess $\mathbf{x}$ by solving the second-order system

$$
\mathbf{H}(\mathbf{x}) \mathbf{d} = -\nabla f(\mathbf{x}),
$$  

where $\mathbf{H}$ is the Hessian matrix, $\nabla f$ is the gradient, and $\mathbf{d}$ is the search direction. The method automatically **augments** the Hessian if it is not positive definite to ensure numerical stability.

---

### Key Components

1. **Symbolic Derivatives (Optional)**  
   - Gradients $\nabla f(\mathbf{x})$ and Hessians $\mathbf{H}(\mathbf{x})$ can be computed symbolically using [SymPy](https://www.sympy.org/).

2. **$\mathbf{LDL}^T$ Factorisation**  
   - Solves linear systems via $\mathbf{LDL}^T$ factorisation.  
   - If $\mathbf{H}$ is indefinite, we add a stabilising shift $a\mathbf{I}$ where:

$$
a = 1.1 \|\mathbf{H}\|_\infty,
$$

and refactor $\mathbf{H}$.

3. **Augmented Hessian for Stability**  
   - Any diagonal entry of $\mathbf{D}$ in the $\mathbf{LDL}^T$ factorisation that is $\leq 0$ triggers augmentation to ensure the system remains solvable.

4. **Convergence Checks**  
   - Stop if:
     - $\|\nabla f(\mathbf{x})\| < \eta$,  
     - $\|\mathbf{d}\| < \epsilon$, or  
     - $|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| < \delta$.

---

### Algorithmic Sketch

1. **Initialisation**:
   - Set $\mathbf{x}_0$ (initial guess), tolerances ($\varepsilon, \delta, \eta$), and maximum iterations (`maxiter`).

2. **Iteration** ($k = 0, 1, \dots$):
   1. Compute $\nabla f(\mathbf{x}_k)$ and $\mathbf{H}(\mathbf{x}_k)$.
   2. Factor $\mathbf{H}(\mathbf{x}_k)$ as $\mathbf{L} \mathbf{D} \mathbf{L}^T$.
   3. If $\mathbf{D}$ has non-positive entries, augment $\mathbf{H}$:

$$
\mathbf{H} \leftarrow \mathbf{H} + a\mathbf{I}, \quad a = 1.1 \|\mathbf{H}\|_\infty.
$$

   4. Solve for $\mathbf{d}$:

$$
\mathbf{H}_{\text{aug}} \mathbf{d} = -\nabla f(\mathbf{x}_k).
$$

   5. Update $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{d}$.
   6. Check convergence (gradient norm, step size, or function difference).
   7. If converged, terminate; otherwise, continue.

3. **Return**:
   - Final solution $\mathbf{x}$, $f(\mathbf{x})$, iteration count, and function evaluations.

---

### When to Use

- **Well-Behaved Hessians**: Performs well when $\mathbf{H}$ is positive definite or only slightly indefinite.  
- **General Smooth Objectives**: Applicable to any differentiable scalar function $f(\mathbf{x})$.  
- **Symbolic or Numerical Derivatives**: Works with both precomputed symbolic derivatives or numerical approximations.

---

### Advantages

- **Fast Quadratic Convergence**: Achieves rapid convergence near the optimum when the Hessian is well-conditioned.  
- **Simple and Efficient**: Minimal overhead compared to more complex trust-region or line-search methods.  
- **Automatic Stabilisation**: Augments the Hessian with $a\mathbf{I}$ to handle indefiniteness, ensuring robustness.

---

### Caveats

- **No Line Search**: Without additional step size control, the method can overshoot or diverge for poorly scaled problems.  
- **Potentially Large Steps**: Indefinite or ill-conditioned Hessians can cause instability without augmentation.  
- **Expensive Factorisation**: Each iteration requires factoring $\mathbf{H}$, which can be computationally expensive for high-dimensional problems.

---

### Example Usage

```python
import sympy as sp
from NewtonRaphson import NewtonRaphsonSolver

# Define a simple function f(x, y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = NewtonRaphsonSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],    # Initial guess
    maxiter=50,       # Maximum iterations
    epsilon=1e-5,     # Step size tolerance
    delta=1e-5,       # Function value tolerance
    eta=1e-5          # Gradient norm tolerance
)

# Solve
result = solver.solve(verbose=True)
print("Solution:", result["x"])
print("Objective value:", result["fun"])
print("Iterations:", result["nit"])
print("Function evaluations:", result["nfev"])
```
