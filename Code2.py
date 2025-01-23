import numpy as np

# -------------------------------------------------------------------
# Utilities: activation, softmax, cross-entropy, etc.
# -------------------------------------------------------------------

def relu_forward(Z):
    """
    Z: shape (batch, channels) or (batch, ...)
    Returns A = ReLU(Z) and cache = Z for backward.
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    dA: gradient wrt output of ReLU
    cache: the Z that produced A
    Returns dZ
    """
    Z = cache
    dZ = (Z > 0) * dA
    return dZ

def softmax_forward(Z):
    """
    Z: (batch, num_classes)
    returns: probabilities P, shape (batch, num_classes)
    """
    # subtract max for numerical stability
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shifted)
    P = expZ / np.sum(expZ, axis=1, keepdims=True)
    return P

def cross_entropy_loss(P, Y_true):
    """
    P: (batch, num_classes) predicted probabilities
    Y_true: (batch, num_classes) one-hot ground truth
    Returns scalar loss, also the derivative dZ4 (i.e. dL/dZ for final layer).
    """
    batch_size = P.shape[0]
    eps = 1e-12
    # cross-entropy
    log_likelihood = -np.log(P + eps) * Y_true
    loss = np.sum(log_likelihood) / batch_size
    # derivative wrt Z (post-softmax), for backprop
    dZ = (P - Y_true) / batch_size
    return loss, dZ

# -------------------------------------------------------------------
# Convolution + Pool (Forward/Backward)
# -------------------------------------------------------------------

def conv_forward(X, W, b):
    """
    Naive 'valid' convolution (no padding, stride=1).
    X shape: (batch, inC, H, W)
    W shape: (outC, inC, kH, kW)
    b shape: (outC,)
    Returns:
      out: (batch, outC, H_out, W_out)
    Also returns a cache for backward.
    H_out = H - kH + 1
    W_out = W - kW + 1
    """
    batch_size, inC, H, W_ = X.shape
    outC, inC2, kH, kW = W.shape
    assert inC == inC2, "Mismatch in channels"
    H_out = H - kH + 1
    W_out = W_ - kW + 1

    out = np.zeros((batch_size, outC, H_out, W_out))
    # naive loops
    for n in range(batch_size):
        for oc in range(outC):
            for i in range(H_out):
                for j in range(W_out):
                    region = X[n, :, i:i+kH, j:j+kW]
                    out[n, oc, i, j] = np.sum(region * W[oc]) + b[oc]

    cache = (X, W, b)
    return out, cache

def conv_backward(dout, cache):
    """
    Backprop for naive conv.
    dout shape: (batch, outC, H_out, W_out)
    cache = (X, W, b)
    Returns dX, dW, db with same shapes as inputs.
    """
    X, W, b = cache
    batch_size, inC, H, W_ = X.shape
    outC, inC2, kH, kW = W.shape
    _, _, H_out, W_out = dout.shape

    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # naive loop
    for n in range(batch_size):
        for oc in range(outC):
            for i in range(H_out):
                for j in range(W_out):
                    val = dout[n, oc, i, j]
                    # For dW
                    region = X[n, :, i:i+kH, j:j+kW]
                    dW[oc] += val * region
                    # For db
                    db[oc] += val
                    # For dX
                    dX[n, :, i:i+kH, j:j+kW] += val * W[oc]

    return dX, dW, db

def maxpool_forward(X):
    """
    2x2 max pool, stride=2
    X shape: (batch, C, H, W)
    Output shape: (batch, C, H//2, W//2)
    We'll store indices for backward.
    """
    batch_size, C, H, W_ = X.shape
    H_out = H // 2
    W_out = W_ // 2
    out = np.zeros((batch_size, C, H_out, W_out))
    # We'll store argmax indices for backward
    idx_mask = np.zeros_like(X, dtype=bool)

    for n in range(batch_size):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_beg = 2*i
                    w_beg = 2*j
                    patch = X[n, c, h_beg:h_beg+2, w_beg:w_beg+2]
                    m = np.max(patch)
                    out[n, c, i, j] = m
                    # find index of max
                    max_loc = np.unravel_index(np.argmax(patch), (2,2))
                    idx_mask[n, c, h_beg+max_loc[0], w_beg+max_loc[1]] = True

    cache = (X, idx_mask)
    return out, cache

def maxpool_backward(dout, cache):
    """
    dOut shape: (batch, C, H_out, W_out)
    cache = (X, idx_mask)
    """
    X, idx_mask = cache
    dX = np.zeros_like(X)
    batch_size, C, H_out, W_out = dout.shape

    for n in range(batch_size):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    val = dout[n, c, i, j]
                    h_beg = 2*i
                    w_beg = 2*j
                    
                    dX[n, c, h_beg:h_beg+2, w_beg:w_beg+2][idx_mask[n, c, h_beg:h_beg+2, w_beg:w_beg+2]] = val

    return dX

# -------------------------------------------------------------------
# Fully-connected (dense) forward/back
# -------------------------------------------------------------------

def fc_forward(X, W, b):
    """
    X: (batch, input_dim)
    W: (output_dim, input_dim)
    b: (output_dim,)
    out: (batch, output_dim)
    """
    out = X @ W.T + b
    cache = (X, W, b)
    return out, cache

def fc_backward(dout, cache):
    """
    dout: (batch, output_dim)
    cache = (X, W, b)
    returns dX, dW, db
    """
    X, W, b = cache
    batch_size, input_dim = X.shape
    output_dim = W.shape[0]

    dX = dout @ W  # shape (batch, input_dim)
    dW = dout.T @ X  # shape (output_dim, input_dim)
    db = np.sum(dout, axis=0)
    return dX, dW, db

# -------------------------------------------------------------------
# Putting it all together in a small training loop
# -------------------------------------------------------------------

def main():
    np.random.seed(42)

    # Hyperparameters
    n1 = 2   # conv1 filters
    n2 = 3   # conv2 filters
    n3 = 5   # hidden units in fc3
    n_out = 10  # output classes
    learning_rate = 1e-2
    num_epochs = 5
    batch_size = 2

    # 1) Initialize random parameters
    # Conv1: (n1, 1, 5, 5)
    W1 = 0.01 * np.random.randn(n1, 1, 5, 5)
    b1 = np.zeros(n1)

    # Conv2: (n2, n1, 5, 5)
    W2 = 0.01 * np.random.randn(n2, n1, 5, 5)
    b2 = np.zeros(n2)

    # fc3: input_dim = (n2*4*4), output_dim = n3
    fc3_in_dim = n2 * 4 * 4
    W3 = 0.01 * np.random.randn(n3, fc3_in_dim)
    b3 = np.zeros(n3)

    # fc4: input_dim = n3, output_dim = n_out
    W4 = 0.01 * np.random.randn(n_out, n3)
    b4 = np.zeros(n_out)

    # 2) Create random "training data"
    # X shape: (batch_size, 1, 28, 28)
    X = np.random.randn(batch_size, 1, 28, 28)
    # Y: one-hot
    Y = np.zeros((batch_size, n_out))
    random_labels = np.random.randint(0, n_out, size=batch_size)
    for i in range(batch_size):
        Y[i, random_labels[i]] = 1.0

    # Training loop
    for epoch in range(num_epochs):
        # -------------------------
        # Forward pass
        # conv1
        Z1, cache1 = conv_forward(X, W1, b1)            # -> (batch,n1,24,24)
        # pool1
        P1, cache_p1 = maxpool_forward(Z1)             # -> (batch,n1,12,12)

        # conv2
        Z2, cache2 = conv_forward(P1, W2, b2)          # -> (batch,n2,8,8)
        # pool2
        P2, cache_p2 = maxpool_forward(Z2)             # -> (batch,n2,4,4)

        # flatten
        batch_size_, C_, H_, W_ = P2.shape
        F = P2.reshape(batch_size_, -1)                # (batch, n2*4*4)

        # fc3 + ReLU
        Z3, cache3 = fc_forward(F, W3, b3)             # (batch, n3)
        A3, relu_cache = relu_forward(Z3)             # (batch, n3)

        # fc4
        Z4, cache4 = fc_forward(A3, W4, b4)            # (batch, n_out)

        # softmax + cross-entropy
        P = softmax_forward(Z4)                       # (batch, n_out)
        loss, dZ4 = cross_entropy_loss(P, Y)

        print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss:.4f}")

        # -------------------------
        # Backward pass
        # fc4
        dA3, dW4, db4 = fc_backward(dZ4, cache4)

        # ReLU
        dZ3 = relu_backward(dA3, relu_cache)

        # fc3
        dF, dW3, db3 = fc_backward(dZ3, cache3)

        # unflatten
        dP2 = dF.reshape(batch_size_, C_, H_, W_)

        # pool2 backward
        dZ2 = maxpool_backward(dP2, cache_p2)

        # conv2 backward
        dP1, dW2, db2 = conv_backward(dZ2, cache2)

        # pool1 backward
        dZ1 = maxpool_backward(dP1, cache_p1)

        # conv1 backward
        dX, dW1, db1 = conv_backward(dZ1, cache1)

        # -------------------------
        # Gradient descent update
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print("Done training.")

if __name__ == "__main__":
    main()
