import numpy as np

def ridge_regression_closed_form(X, y, alpha):
    """
    Closed-form solution for ridge regression:
    w* = (X^T X + alpha I)^{-1} X^T y
    
    Parameters:
    -----------
    X     : (n, d) array, design matrix
    y     : (n,)   array, target values
    alpha : float, regularization coefficient
    
    Returns:
    --------
    w_cf  : (d,)   array, the solution vector
    """
    n, d = X.shape
    # (X^T X + alpha I)
    A = X.T @ X + alpha * np.eye(d)
    b = X.T @ y
    # Solve A w_cf = b
    w_cf = np.linalg.inv(A) @ b
    return w_cf

def ridge_regression_gradient_descent(X, y, alpha, lr=1e-3, max_iter=1000, tol=1e-6):
    """
    Gradient Descent for ridge regression:
    Minimize L(w) = ||y - Xw||^2 + alpha ||w||^2
    
    Parameters:
    -----------
    X        : (n, d) array, design matrix
    y        : (n,)   array, target values
    alpha    : float, regularization coefficient
    lr       : float, learning rate
    max_iter : int,   maximum number of iterations
    tol      : float, tolerance for stopping
    
    Returns:
    --------
    w_gd : (d,) array, the solution vector after gradient descent
    """
    n, d = X.shape
    
    # Initialize weights (e.g., zeros or small random)
    w_gd = np.zeros(d)
    
    for i in range(max_iter):
        # Gradient = 2 (X^T X w - X^T y) + 2 alpha w
        # (for L(w) = (y - Xw)^T (y - Xw) + alpha w^T w)
        
        # 1) Compute predictions
        y_pred = X @ w_gd
        # 2) Residual
        r = y_pred - y
        # 3) Gradient
        grad = 2 * (X.T @ r) + 2 * alpha * w_gd
        
        # Update step
        w_new = w_gd - lr * grad
        
        # Check for convergence
        if np.linalg.norm(w_new - w_gd) < tol:
            w_gd = w_new
            break
        
        w_gd = w_new
    
    return w_gd

def ridge_regression_newton_method(X, y, alpha, max_iter=10, tol=1e-9):
    """
    Newton's Method for ridge regression:
    L(w) = ||y - Xw||^2 + alpha ||w||^2
    
    - Grad(L) = 2(X^T X w - X^T y) + 2 alpha w
    - Hess(L) = 2(X^T X + alpha I)
    
    Newton update:
    w_{new} = w - H^{-1} grad
    
    For a strictly quadratic problem (like ridge),
    Newton's method converges in one step if started from w=0.
    But we show the iterative form for completeness.
    
    Parameters:
    -----------
    X        : (n, d) array
    y        : (n,)   array
    alpha    : float
    max_iter : int
    tol      : float
    
    Returns:
    --------
    w_nt : (d,) array, solution vector
    """
    n, d = X.shape
    w_nt = np.zeros(d)
    
    # Precompute Hessian = 2 (X^T X + alpha I)
    H = 2.0 * (X.T @ X + alpha * np.eye(d))
    
    for i in range(max_iter):
        # grad = 2 (X^T X w_nt - X^T y) + 2 alpha w_nt
        y_pred = X @ w_nt
        r = y_pred - y
        grad = 2 * (X.T @ r) + 2 * alpha * w_nt
        
        # Newton step
        delta = np.linalg.inv(H) @ grad
        w_new = w_nt - delta
        
        # Check convergence
        if np.linalg.norm(w_new - w_nt) < tol:
            w_nt = w_new
            break
        
        w_nt = w_new
    
    return w_nt

# -----------------------
# Example usage (test):
if __name__ == "__main__":
    # Generate a small synthetic data set
    np.random.seed(42)
    n, d = 50, 3
    X = np.random.randn(n, d)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(n)  # small noise
    
    alpha = 1.0  # regularization
    
    w_cf = ridge_regression_closed_form(X, y, alpha)
    w_gd = ridge_regression_gradient_descent(X, y, alpha, lr=1e-2, max_iter=5000)
    w_nt = ridge_regression_newton_method(X, y, alpha)
    
    print("Closed-form solution :", w_cf)
    print("Gradient descent     :", w_gd)
    print("Newton's method      :", w_nt)
