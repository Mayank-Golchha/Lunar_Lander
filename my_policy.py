import numpy as np

def policy_action(policy, observation):
    idx = 0

    W1 = policy[idx: idx + 8 * 16].reshape(8, 16)
    idx += 8 * 16

    b1 = policy[idx: idx + 16]
    idx += 16

    W2 = policy[idx: idx + 16 * 4].reshape(16, 4)
    idx += 16 * 4

    b2 = policy[idx: idx + 4]

    x = np.array(observation, dtype=np.float32)
    hidden = np.tanh(np.dot(x, W1) + b1)
    logits = np.dot(hidden, W2) + b2

    return int(np.argmax(logits))