import numpy as np

def policy_action(params, observation):
    # Network architecture: 8 -> 16 -> 4
    # Total params: (8*16 + 16) + (16*4 + 4) = 144 + 68 = 212

    idx = 0

    W1 = params[idx: idx + 8 * 16].reshape(8, 16)
    idx += 8 * 16

    b1 = params[idx: idx + 16]
    idx += 16

    W2 = params[idx: idx + 16 * 4].reshape(16, 4)
    idx += 16 * 4

    b2 = params[idx: idx + 4]

    # Forward pass
    hidden = np.tanh(np.dot(observation, W1) + b1)
    logits = np.dot(hidden, W2) + b2

    return np.argmax(logits)