
---

## **Unit 1: Solution of Nonlinear Equations**

1. **Q:** *What is a nonlinear equation? Provide examples.*  
   **A:** A nonlinear equation is one in which the unknown variable appears with exponents other than one, or inside functions like exponentials, logarithms, or trigonometric functions. Examples include:  
   - \( x^2 - 4 = 0 \)  
   - \( e^x - 2 = 0 \)  
   - \( \sin(x) - \frac{x}{2} = 0 \)

2. **Q:** *Describe the Bisection Method and its convergence criteria.*  
   **A:** The Bisection Method is a bracketing method that starts with an interval \([a, b]\) where \( f(a) \) and \( f(b) \) have opposite signs (i.e., \( f(a)f(b) < 0 \)). It computes the midpoint \( c = \frac{a+b}{2} \) and replaces one end of the interval with \( c \) based on the sign of \( f(c) \). This process repeats until the interval is sufficiently small. Convergence is guaranteed if the function is continuous on \([a, b]\) and there is a sign change at the endpoints.

3. **Q:** *How does the False Position (Regula Falsi) Method differ from the Bisection Method?*  
   **A:** While both methods require an initial bracket where the function changes sign, the False Position Method uses linear interpolation between the endpoints to estimate the root, potentially leading to faster convergence. However, it may suffer from stagnation if one endpoint does not move.

4. **Q:** *Explain the Newton-Raphson Method and discuss its advantages and limitations.*  
   **A:** The Newton-Raphson Method uses the formula  
   \[
   x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
   \]  
   to iteratively approximate a root. Its advantages include rapid (quadratic) convergence when the initial guess is close to the actual root. Limitations include the need for the derivative \( f'(x) \) and possible divergence if the initial guess is poor or if \( f'(x) \) is zero or near zero.

5. **Q:** *What is fixed point iteration and what condition ensures its convergence?*  
   **A:** Fixed point iteration rewrites an equation as \( x = g(x) \) and computes successive approximations via \( x_{n+1} = g(x_n) \). Convergence is ensured if \( g(x) \) is a contraction on the interval, meaning there exists a constant \( L \) with \( 0 \leq L < 1 \) such that \( |g'(x)| \leq L \) for all \( x \) in the interval.

6. **Q:** *How can a system of nonlinear equations be solved using Newton's Method?*  
   **A:** For a system \( \mathbf{F}(\mathbf{x}) = \mathbf{0} \), Newton's method uses the Jacobian matrix \( J(\mathbf{x}) \) of partial derivatives. The iterative update is given by  
   \[
   \mathbf{x}_{n+1} = \mathbf{x}_n - J^{-1}(\mathbf{x}_n)\mathbf{F}(\mathbf{x}_n)
   \]  
   requiring the solution of a linear system at each step.

7. **Q:** *What types of errors can occur when solving nonlinear equations numerically?*  
   **A:** Common errors include **truncation error** (due to approximating an infinite process with a finite number of steps) and **round-off error** (due to limited precision in computer arithmetic). Additionally, improper initial guesses may lead to divergence or convergence to an unintended root.

---

## **Unit 2: Interpolation and Approximation**

1. **Q:** *What is polynomial interpolation?*  
   **A:** Polynomial interpolation is the process of finding a polynomial \( P(x) \) of degree \( n \) that passes exactly through \( n+1 \) given data points. It is used to estimate unknown values of a function based on known data.

2. **Q:** *Explain Lagrange’s interpolation formula.*  
   **A:** Lagrange’s interpolation formula constructs the interpolating polynomial as:  
   \[
   P(x) = \sum_{i=0}^{n} y_i L_i(x)
   \]  
   where the Lagrange basis polynomials \( L_i(x) \) are defined as:  
   \[
   L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}
   \]  
   Each \( L_i(x) \) equals 1 at \( x_i \) and 0 at all other data points.

3. **Q:** *What are divided differences and how are they used in Newton's interpolation?*  
   **A:** Divided differences are recursive coefficients computed from the data points. In Newton's interpolation, the polynomial is written as:  
   \[
   P(x) = f[x_0] + f[x_0, x_1](x - x_0) + f[x_0, x_1, x_2](x - x_0)(x - x_1) + \cdots
   \]  
   They allow for the efficient addition of new data points and provide an alternative formulation to Lagrange's method.

4. **Q:** *What is cubic spline interpolation and why is it used?*  
   **A:** Cubic spline interpolation uses piecewise cubic polynomials between data points (knots) while ensuring that the first and second derivatives are continuous at the knots. This method produces a smooth curve that avoids the oscillations common in high-degree polynomial interpolation.

5. **Q:** *Describe the Least Squares Method for data approximation.*  
   **A:** The Least Squares Method finds an approximate solution by minimizing the sum of the squares of the differences between the observed values and the values predicted by a model. It is especially useful when the data contains errors or noise and an exact fit is not achievable.

6. **Q:** *What are the primary sources of error in polynomial interpolation?*  
   **A:** Errors in polynomial interpolation may arise due to:
   - **Runge’s Phenomenon:** Oscillations at the edges of the interval for high-degree polynomials.
   - **Numerical Instability:** Errors due to the large condition numbers when using high-degree polynomials.
   - **Distribution of Points:** Poor choice of interpolation points can lead to larger errors.

---

## **Unit 3: Numerical Differentiation and Integration**

1. **Q:** *What is numerical differentiation and what are Newton’s differentiation formulas?*  
   **A:** Numerical differentiation approximates derivatives using discrete data. Newton’s differentiation formulas include:
   - **Forward Difference:**  
     \[
     f'(x) \approx \frac{f(x+h) - f(x)}{h}
     \]
   - **Backward Difference:**  
     \[
     f'(x) \approx \frac{f(x) - f(x-h)}{h}
     \]
   - **Central Difference:**  
     \[
     f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
     \]
   These formulas use function values at nearby points to estimate the derivative.

2. **Q:** *Explain the Trapezoidal Rule for numerical integration.*  
   **A:** The Trapezoidal Rule approximates the integral by dividing the interval \([a, b]\) into subintervals and approximating the area under the curve as trapezoids. For a single subinterval, it is given by:  
   \[
   \int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(a) + f(b)\right]
   \]  
   and for multiple subintervals, the formula is applied to each segment and summed.

3. **Q:** *Describe Simpson’s 1/3 Rule and its requirements.*  
   **A:** Simpson’s 1/3 Rule approximates the integral by fitting a quadratic polynomial through three points. The formula for two subintervals is:  
   \[
   \int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + f(x_2)\right]
   \]  
   It requires that the number of subintervals is even.

4. **Q:** *What is Simpson’s 3/8 Rule?*  
   **A:** Simpson’s 3/8 Rule uses cubic interpolation over three subintervals. Its formula is:  
   \[
   \int_a^b f(x)\,dx \approx \frac{3h}{8}\left[f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3)\right]
   \]  
   It is applied when the number of subintervals is a multiple of three.

5. **Q:** *How does Romberg Integration work?*  
   **A:** Romberg Integration improves the Trapezoidal Rule by using Richardson extrapolation. It constructs a tableau of estimates with successively halved step sizes and extrapolates these to obtain a highly accurate estimate of the integral.

6. **Q:** *What are common error sources in numerical differentiation and integration?*  
   **A:** The main error sources are:
   - **Truncation Error:** From approximating the true derivative or integral with a finite difference or formula.
   - **Round-Off Error:** Due to the finite precision of computer arithmetic.
   Choosing an appropriate step size \( h \) is critical to balance these errors.

7. **Q:** *Describe numerical double integration.*  
   **A:** Numerical double integration extends single-variable techniques to two dimensions. Methods such as the double trapezoidal or Simpson’s rule are applied iteratively over a grid to approximate the integral over a two-dimensional domain.

---

## **Unit 4: Solution of Linear Algebraic Equations**

1. **Q:** *How do you determine if a system of linear equations is consistent?*  
   **A:** A system is consistent if it has at least one solution. This is determined by converting the system into an augmented matrix and checking for rows that would represent an impossibility (e.g., \( 0 = \text{nonzero} \)). If no such row exists, the system is consistent.

2. **Q:** *Explain the Gaussian Elimination Method.*  
   **A:** Gaussian elimination systematically applies row operations to transform a system of equations into an upper triangular form. Once in this form, back-substitution is used to solve for the unknown variables.

3. **Q:** *What is Gauss-Jordan elimination and how does it differ from Gaussian elimination?*  
   **A:** Gauss-Jordan elimination continues the process of Gaussian elimination to transform the matrix into reduced row-echelon form (diagonal form), where each leading coefficient is 1 and is the only nonzero entry in its column. This method yields the solution directly without the need for back-substitution.

4. **Q:** *How can the inverse of a matrix be computed using Gaussian elimination?*  
   **A:** To compute \( A^{-1} \), form the augmented matrix \([A \, | \, I]\) and apply row operations to reduce \( A \) to the identity matrix. The transformed right-hand side becomes the inverse \( A^{-1} \).

5. **Q:** *Describe the iterative methods—Jacobi and Gauss-Seidel—for solving linear systems.*  
   **A:**  
   - **Jacobi Method:** Updates all variables simultaneously using values from the previous iteration.  
   - **Gauss-Seidel Method:** Updates variables sequentially, using the most recent values immediately in the computation.  
   Both methods require conditions (such as diagonal dominance) for convergence.

6. **Q:** *Explain the Power Method for finding eigenvalues.*  
   **A:** The Power Method is an iterative algorithm used to approximate the dominant eigenvalue (largest in magnitude) and its eigenvector. Starting with an initial guess vector, the matrix is repeatedly applied and the result normalized until convergence is achieved.

---

## **Unit 5: Solution of Ordinary Differential Equations**

1. **Q:** *What is an initial value problem (IVP) in the context of ODEs?*  
   **A:** An IVP involves a differential equation along with specified initial conditions. For example, given \( y' = f(x,y) \) and an initial condition \( y(x_0) = y_0 \), the task is to find \( y(x) \) that satisfies both the differential equation and the initial condition.

2. **Q:** *Describe Euler’s Method and its limitations regarding accuracy.*  
   **A:** Euler’s Method approximates the solution using:  
   \[
   y_{n+1} = y_n + h\, f(x_n, y_n)
   \]  
   It is simple but only first-order accurate, meaning the local truncation error is \( O(h^2) \) and the global error is \( O(h) \). This can lead to significant inaccuracies unless very small step sizes are used.

3. **Q:** *How does the Runge-Kutta method improve upon Euler’s Method?*  
   **A:** Runge-Kutta methods (especially the fourth-order Runge-Kutta) compute intermediate slopes within each step and combine them in a weighted average to achieve a much higher order of accuracy (global error \( O(h^4) \)) without requiring extremely small step sizes.

4. **Q:** *What is Picard’s Method?*  
   **A:** Picard’s Method, also known as the method of successive approximations, uses the integral form of the ODE:  
   \[
   y(x) = y_0 + \int_{x_0}^{x} f(t, y(t))\,dt
   \]  
   It generates successive approximations that converge to the actual solution under appropriate conditions.

5. **Q:** *Explain the Shooting Method for solving boundary value problems (BVPs).*  
   **A:** The Shooting Method converts a BVP into an IVP by guessing the missing initial condition (often the slope) and integrating to the other boundary. The guess is iteratively adjusted until the computed solution satisfies the boundary condition at the other end.

6. **Q:** *How can a higher-order ODE be converted into a system of first-order ODEs?*  
   **A:** A higher-order ODE can be rewritten by defining new variables for each derivative up to one order less than the original. For example, an \( n \)th order ODE can be expressed as a system of \( n \) first-order ODEs, which is often easier to solve numerically.

7. **Q:** *What is the Taylor Series Method for solving ODEs?*  
   **A:** The Taylor Series Method involves expanding the solution \( y(x) \) in a Taylor series about a point and using the differential equation to compute the necessary derivatives. Including more terms in the series generally improves the accuracy of the approximation.

---

## **Unit 6: Solution of Partial Differential Equations**

1. **Q:** *What is a partial differential equation (PDE)?*  
   **A:** A PDE involves partial derivatives of a function with respect to multiple independent variables. Examples include the heat equation, wave equation, and Laplace’s equation, each modeling different physical phenomena.

2. **Q:** *How are finite difference approximations used to solve PDEs?*  
   **A:** Finite difference methods replace derivatives in a PDE with difference quotients using values at discrete grid points. This converts the PDE into a system of algebraic equations that can be solved numerically.

3. **Q:** *Describe the Laplace Equation and its importance.*  
   **A:** The Laplace Equation,  
   \[
   \nabla^2 u = 0,
   \]  
   is a second-order PDE that models steady-state phenomena such as electrostatics, heat conduction, and incompressible fluid flow. Its solutions, called harmonic functions, have properties like the mean value property and are free of local extrema in the interior of a domain.

4. **Q:** *What is Poisson’s Equation and how does it differ from the Laplace Equation?*  
   **A:** Poisson’s Equation is written as:  
   \[
   \nabla^2 u = f,
   \]  
   where \( f \) represents a source term (such as a charge distribution in electrostatics). When \( f = 0 \), Poisson’s Equation reduces to the Laplace Equation.

5. **Q:** *What types of boundary conditions are typically applied when solving Laplace’s and Poisson’s equations?*  
   **A:** Common boundary conditions include:  
   - **Dirichlet Boundary Conditions:** Specifying the value of the function on the boundary.  
   - **Neumann Boundary Conditions:** Specifying the value of the derivative (flux) normal to the boundary.  
   - **Mixed Boundary Conditions:** A combination of Dirichlet and Neumann conditions, depending on the physical problem.

---
