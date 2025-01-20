# Nonlinear Optimisation - Finding local minimisers

----



----


## 1. Trust-Region-Style [Levenberg–Marquardt Method](Classes/LevenbergMarquardt.py)

Python implementation of a **trust-region–style** Levenberg–Marquardt algorithm, which adaptively modifies the Hessian by adding a diagonal shift to ensure positive definiteness. This approach is useful for solving non-linear least squares or general unconstrained optimisation problems and helps handle situations where the Hessian is indefinite or near-singular.

**Levenberg–Marquardt (LM)** is a blend of **Newton’s method** and **gradient descent**, originally popular in non-linear least squares contexts. In classical LM, we solve

$$
(\mathbf{H} + \lambda \mathbf{I}) \, \mathbf{d} = -\nabla f,
$$

where $\mathbf{H}$ is the Hessian and $\lambda$ is a *damping* parameter. By adjusting $\lambda$, we move between small “gradient descent–like” steps when the function is difficult (large $\lambda$) and near-Newton steps when we’re close to the solution (small $\lambda$).

**Trust-region–style acceptance** is a way to decide if we “trust” the proposed step $d$. Instead of doing a separate line search, we compare:
- The **actual reduction** in function value:

$$
\text{actual} = f(\mathbf{x}) - f(\mathbf{x} + \mathbf{d})
$$

- The **predicted reduction** from the local quadratic model:

$$
\text{predicted} 
    = -\,\nabla f(x)^T \mathbf{d} 
      \;-\;\tfrac12\,\mathbf{d}^T \mathbf{H}_{\text{model}}\,\mathbf{d},
$$

where $\mathbf{H}_{\text{model}}$ can be $\mathbf{H} + \lambda \mathbf{I}$ or just $\mathbf{H}$.

We form the ratio

$$
\rho = \frac{\text{actual}}{\text{predicted}}
$$

- If $\rho$ is large ($> 0.75$), the step was better (or about as good) as predicted, so we *accept* it and *decrease* $\lambda$.  
- If $\rho$ is small ($< 0.25$), the step did not reduce the function as much as expected (or at all), so we *reject* it, keep the old $x$, and *increase* $\lambda$. This “shrinks the trust-region” to take a safer, smaller step next time.

----

### Key Components

1. **Symbolic Derivatives (Optional)**  
   - We use [SymPy](https://www.sympy.org/) to generate lambdified functions for the gradient and Hessian. This makes it easier to ensure correct derivatives.

2. **$\mathbf{LDL}^T$ Decomposition**  
   - We factor $(\mathbf{H} + a\mathbf{I})$ via an [$\mathbf{LDL}^T$ decomposition]([ldl_decomposition.py](Procedures/ldl_decomposition.py)). 
   - The shift $a$ ensures the matrix is positive definite if $\lambda$ is chosen sufficiently large.

3. **Adaptive Lambda**  
   - We set $a = \lambda \cdot \|\mathbf{H}\|_\infty$ to scale the shift by the Hessian’s magnitude. This helps with poorly scaled problems where a fixed $\lambda$ may be too big or too small.
   - We update $\lambda$ according to how successful the step is ($\rho$).

4. **Acceptance or Rejection**  
   - We compute $\rho$. If $\rho < 0.25$, we reject and increase $\lambda$. If $\rho > 0.75$, we decrease $\lambda$. Otherwise, we leave $\lambda$ unchanged.

5. **Convergence Checks**  
   - We check standard criteria:
     - Gradient norm $\|\nabla f(\mathbf{x})\|$ below a tolerance
     - Step size $\|\mathbf{d}\|$ below a tolerance
     - Actual reduction $|f(\mathbf{x+d}) - f(\mathbf{x})|$ below a tolerance

If these criteria are satisfied, we conclude that the method has converged.

----

### Pseudocode

Below is a high-level sketch of the algorithm:

1. **Initialize**  
   - $x_0$ as the initial guess  
   - $\lambda$ (damping parameter)  
   - tolerances: $\epsilon,\ \delta,\ \eta$  
   - maximum iterations: `maxiter`

2. **Loop** over $k$ in $1,\dots,\text{maxiter}$:
   1. Compute $\nabla f(x_k)$ and $H(x_k)$.  
   2. Form $H_{\text{aug}} = H(x_k) + a I$ where $a = \lambda \cdot \|H(x_k)\|_\infty$.  
   3. Factor $H_{\text{aug}}$ (LDLᵀ or Cholesky).  
   4. Solve $H_{\text{aug}} \, d = -\nabla f(x_k)$.  
   5. Compute **predicted reduction**.  
   6. Evaluate $f(x_k + d)$ → **actual reduction**.  
   7. Compute $\rho = \frac{\text{actual}}{\text{predicted}}$.  
   8. If $\rho < 0.25$, *reject step*, increase $\lambda$.  
   9. Else *accept step* ($x_{k+1} \leftarrow x_k + d$) and if $\rho > 0.75$, reduce $\lambda$.  
   10. Check stopping criteria (gradient norm, step size, function reduction). If satisfied, stop.

3. Return results (best $x$, final function value, iteration count, etc.).

----

### When to Use

- **Nonlinear Least Squares**: Standard LM is classically used in curve-fitting or residual-minimisation tasks.  
- **General Smooth Objectives**: This version can handle any smooth function $f$ because we form its Hessian explicitly.  
- **Ill-Conditioned Hessians**: Because of the adaptive damping, it can handle near-singular Hessians better than a pure Newton method.

### Advantages

- **Robustness**: The trust-region aspect (rejecting poor steps, shrinking the effective region) helps prevent large, unwarranted steps.  
- **Adaptive Damping**: By adjusting $\lambda$ based on success or failure of each step, we smoothly transition between quasi-Newton steps (small $\lambda$) and gradient-descent-like steps (large $\lambda$) when necessary.  
- **Scale Sensitivity**: Multiplying $\lambda$ by $\|\mathbf{H}\|_\infty$ means the damping magnitude is automatically scaled to the problem’s local curvature.

### Caveats

- **Symbolic Overhead**: Generating or evaluating symbolic Hessians may be expensive for high-dimensional problems.  
- **LDLᵀ Factorization**: If the matrix is still indefinite despite the shift, you may need a “retry” loop that continues doubling $\lambda$.  
- **No Separate Line Search**: In trust-region LM, the “actual vs. predicted” check supersedes the need for a typical line search, but you can combine them at extra complexity if desired.

### Example Usage

```python
from lm_solver import LevenbergMarquardtSolver
import sympy as sp

# Define a simple function f(x,y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = LevenbergMarquardtSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],      # initial guess
    lamb=1.0,           # initial damping parameter
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

## 2. Backtracking-Line-Search Style [Levenberg-Marquardt Method](Classes/LevenbergMarquardtLineSearch.py)

Python implementation of a **hybrid** Levenberg–Marquardt method that 
1. forms a damped Newton step using 
$$
(\mathbf{H} + \lambda \mathbf{I})\, \mathbf{d} = -\nabla f
$$
2. applies a backtracking line search to refine the step length. This approach helps handle indefiniteness in the Hessian (via damping) while also preventing overly large steps using an **Armijo** condition.

**Levenberg–Marquardt (LM) + Line Search** combines:
1. A **damping** term $\lambda \mathbf{I}$ to ensure positive definiteness or mitigate ill-conditioning of $\mathbf{H}$.
2. A **backtracking search** to find a suitable step length $\alpha$ that satisfies a sufficient decrease condition.

In each iteration:

- We compute the Hessian $ \mathbf{H}(\mathbf{x}) $ and gradient $ \nabla f(\mathbf{x}) $.
- Form the “augmented” Hessian 
$$
\mathbf{H}_{\text{aug}} \;=\; \mathbf{H}(\mathbf{x}) \;+\; a\,\mathbf{I},
$$
where $ a = \lambda \,\|\mathbf{H}(\mathbf{x})\|_\infty $.
- Solve 
$$
\mathbf{H}_{\text{aug}}\,\mathbf{d} \;=\; -\nabla f(\mathbf{x}).
$$
- Use **backtracking** to find $\alpha \in [0,1]$ such that
$$
f(\mathbf{x} + \alpha\,\mathbf{d}) \;\le\; f(\mathbf{x}) \;+\; c\,\alpha\, \nabla f(\mathbf{x})^T \mathbf{d}.
$$
- If the line search **fails** to find improvement, we **reject** the step, **increase** $\lambda$, and re-try. If it **succeeds**, we **accept** the step, possibly **decrease** $\lambda$, and move on.

----

### Key Components

1. **Symbolic Derivatives**  
   - We optionally use [SymPy](https://www.sympy.org/) to auto-generate the gradient $ \nabla f(\mathbf{x}) $ and Hessian $ \mathbf{H}(\mathbf{x}) $.  

2. **$\mathbf{LDL}^T$ Decomposition**  
   - We factor $ \mathbf{H}_{\text{aug}} = \mathbf{H} + a \mathbf{I} $ via an [$\mathbf{LDL}^T$ decomposition]([ldl_decomposition.py](Procedures/ldl_decomposition.py)), ensuring numerical stability if $ \mathbf{H}_{\text{aug}} $ is positive definite.

3. **Backtracking Line Search**  
   - Implements **Armijo** sufficient-decrease condition:
     $$
     f(\mathbf{x} + \alpha\,\mathbf{d}) \;\le\; f(\mathbf{x}) \;+\; c\,\alpha\, \nabla f(\mathbf{x})^T \mathbf{d}.
     $$
   - If no $\alpha$ satisfies this within a few attempts, the step is “rejected” and damping $\lambda$ is increased.

4. **Adaptive Damping**  
   - After a successful step, we typically do $ \lambda \leftarrow 0.8\,\lambda $.  
   - If no improvement occurs, $ \lambda $ is **doubled** and we re-factor $ \mathbf{H} + \lambda \mathbf{I} $.

5. **Convergence Checks**  
   - Stop if the gradient norm $ \|\nabla f(\mathbf{x})\| $ is small enough, the step size $ \|\mathbf{d}\| $ is small enough, or the function value changes below a tolerance.

----

### Pseudocode

1. **Initialization**  
   - $ \mathbf{x}_0 $: initial guess  
   - $ \lambda $: damping parameter  
   - Tolerances: $ \epsilon, \delta, \eta $  
   - `maxiter`: max iterations  

2. **Iteration** for $ k $ in $ 1,2,\dots $ up to `maxiter`:
   1. Evaluate $ \nabla f(\mathbf{x}_k) $ and $ \mathbf{H}(\mathbf{x}_k) $.  
   2. Form 
   $$
   \mathbf{H}_{\text{aug}} \;=\; \mathbf{H}(\mathbf{x}_k) \;+\; \lambda \,\|\mathbf{H}(\mathbf{x}_k)\|_\infty \,\mathbf{I}.
   $$
   3. Factor $ \mathbf{H}_{\text{aug}} $ (via LDLᵀ).  
   4. Solve 
   $$
   \mathbf{H}_{\text{aug}}\,\mathbf{d} \;=\; -\,\nabla f(\mathbf{x}_k).
   $$
   5. **Backtracking search**: Find $ \alpha \le 1 $ s.t. Armijo holds.  
   6. If no such $ \alpha $ is found, **increase** $ \lambda $ and retry from step 2.  
   7. Otherwise, compute $ f(\mathbf{x}_k + \alpha\,\mathbf{d}) $. If it’s an improvement:
      - Accept the step: $ \mathbf{x}_{k+1} \leftarrow \mathbf{x}_k + \alpha\,\mathbf{d} $.  
      - Possibly **decrease** $ \lambda $.  
   8. Check convergence (gradient norm, step size, function change).  

3. **Return** $(\mathbf{x}, f(\mathbf{x}), \text{iterations}, \text{nfev})$.

----

### When to Use

- **Ill-Conditioned Problems**: Damping stabilizes the Newton step if $ \mathbf{H} $ is indefinite or near-singular.  
- **Desire for a Smaller Step**: The line search can reduce step lengths that would otherwise be too large or skip over minima.  
- **General Smooth Objectives**: Works for any scalar function $ f $; not restricted to least squares.

### Advantages

- **Combines Damping + Line Search**: Avoids large or unproductive steps by regulating both direction (via $ \lambda $) and step length (via Armijo).  
- **Adaptive**: Increases $ \lambda $ if no improvement is found, decreases $ \lambda $ if we succeed.  
- **Symbolic Differentiation**: Minimizes risk of coding gradient/Hessian incorrectly.

### Caveats

- **Repeated Factorizations**: If the step fails, we re-factor $ \mathbf{H} + \lambda \mathbf{I} $ with a larger $ \lambda $.  
- **Extra Function Evaluations**: The line search calls $ f $ repeatedly, which can be expensive if $ f $ is complex.  
- **Tuning**: The constants (0.5 for backtracking, 0.8 for reducing $ \lambda $, etc.) can be adjusted for performance.

----

### Example Usage

```python
import sympy as sp
from Classes.LevenbergMarquardt import LevenbergMarquardtSolver

# Define a function f(x,y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = LevenbergMarquardtSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],    # initial guess
    lamb=1.0,         # initial damping
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

## 3. [Newton–Raphson Method](Classes/NewtonRaphson.py)

This Python implementation uses **Newton’s method** with an $ \mathbf{LDL}^T $ factorization to solve  
$$
\mathbf{H}(\mathbf{x}) \,\mathbf{d} = -\,\nabla f(\mathbf{x}),
$$  
and **augments** the Hessian if $ \mathbf{H} $ is not positive definite.

**Newton–Raphson method** iteratively refines a guess $ \mathbf{x} $ by using the local second-order (Hessian) approximation of the function $ f $.

In each iteration:
1. We compute $ \nabla f(\mathbf{x}) $ and $ \mathbf{H}(\mathbf{x}) $.  
2. Factor $ \mathbf{H}(\mathbf{x}) $ via $ \mathbf{LDL}^T $.  
3. If $ \mathbf{D} $ (the diagonal in $ \mathbf{LDL}^T $) has any non-positive entries, we add $ a\mathbf{I} $ to $ \mathbf{H} $ (with $ a $ scaled by $ \| \mathbf{H} \|_\infty $) and refactor.  
4. Solve for $ \mathbf{d} $ in
   $$
   (\mathbf{H}(\mathbf{x}) + a\,\mathbf{I})\,\mathbf{d} = -\,\nabla f(\mathbf{x}).
   $$
5. Update $ \mathbf{x} \leftarrow \mathbf{x} + \mathbf{d} $.  
6. Check convergence criteria (gradient norm, step size, function difference).

----

### Key Components

1. **Symbolic Derivatives (Optional)**  
   - We can use [SymPy](https://www.sympy.org/) to generate $ \nabla f(\mathbf{x}) $ and $ \mathbf{H}(\mathbf{x}) $ automatically.

2. **$\mathbf{LDL}^T$ Factorization**  
   - We use [$\mathbf{LDL}^T$ decomposition]([ldl_decomposition.py](Procedures/ldl_decomposition.py)) to solve linear systems with $ \mathbf{H}(\mathbf{x}) $.  
   - If $ \mathbf{H} $ is not positive definite, we shift it by $ a\mathbf{I} $, ensuring the factorization is stable.

3. **Augmented Hessian for Non-Positive $ \mathbf{D} $**  
   - When any diagonal element of $ \mathbf{D} $ is $ \le 0 $, we add 
   $$
   a = 1.1 \,\|\mathbf{H}\|_\infty
   $$  
   to $ \mathbf{H} $ and refactor.

4. **Convergence Checks**  
   - Stop if  
     - $ \|\nabla f(\mathbf{x})\| \lt \eta $  
     - $ \|\mathbf{x}_{k+1} - \mathbf{x}_k\| \lt \epsilon $  
     - $ |f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| \lt \delta $  

----

### Algorithmic Sketch

1. **Initialize**  
   - $ \mathbf{x}_0 $ as initial guess  
   - Tolerances: $ \epsilon, \delta, \eta $  
   - Max iterations: `maxiter`

2. **Iteration** ($ k = 0,1,\dots $):
   1. Compute $ \nabla f(\mathbf{x}_k) $, $ \mathbf{H}(\mathbf{x}_k) $.  
   2. Factor $ \mathbf{H}(\mathbf{x}_k) = \mathbf{L}\,\mathbf{D}\,\mathbf{L}^T $.  
   3. If $ \mathbf{D} $ has non-positive entries, set 
   $$
   \mathbf{H} \leftarrow \mathbf{H} + a\,\mathbf{I}, \quad a = 1.1\,\|\mathbf{H}\|_\infty,
   $$
   then refactor.  
   4. Solve  
   $$
   \mathbf{H}(\mathbf{x}_k)\,\mathbf{d} = -\,\nabla f(\mathbf{x}_k).
   $$
   5. Update $ \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{d} $.  
   6. Check if $ \|\nabla f(\mathbf{x}_{k+1})\| \lt \eta $ etc.  
   7. If converged, stop; else continue.

3. **Return** $ (\mathbf{x}, f(\mathbf{x}), \text{iterations}, \text{nfev}) $.

----

### When to Use

- **Well-Behaved or Mildly Indefinite Hessians**: A “plain” Newton step is typically fast if $ \mathbf{H} $ is fairly well-conditioned or only slightly indefinite.  
- **General Smooth Objectives**: Works for generic scalar functions $ f(\mathbf{x}) $.  
- **Symbolic or Numerical Derivatives**: Flexible whether you compute derivatives symbolically or approximate them numerically.

### Advantages

- **Fast Quadratic Convergence**: Newton’s method is often very efficient near the optimum.  
- **Simple**: Minimal overhead compared to trust-region or line-search methods.  
- **Automatic Hessian Fix**: If the Hessian is indefinite, adding $ a\mathbf{I} $ easily stabilizes the step.

### Caveats

- **No Line Search**: If the Hessian or update is poor, steps can overshoot. (Though we do add $ a\mathbf{I} $ if needed to fix indefiniteness.)  
- **Potential Large Steps**: Without damping or step control, the algorithm can diverge on certain non-convex problems.  
- **Expensive Factorization**: Each iteration requires factoring $ \mathbf{H} $.

----

### Example Usage

```python
import sympy as sp
from Classes.NewtonRaphson import NewtonRaphsonSolver

# Define a simple function f(x,y) = (x - 2)^2 + (y + 1)^2
x, y = sp.symbols('x y', real=True)
f_expr = (x - 2)**2 + (y + 1)**2

# Create the solver
solver = NewtonRaphsonSolver(
    f=f_expr,
    variables=[x, y],
    x0=[0.0, 0.0],    # initial guess
    maxiter=50
)

# Solve
result = solver.solve(verbose=True)
print("Solution:", result["x"])
print("Objective value:", result["fun"])
print("Iterations:", result["nit"])
print("Function evaluations:", result["nfev"])
```

