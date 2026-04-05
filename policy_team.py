import numpy as np

# Network architecture constants
INPUT_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 4


def get_policy_size():
    """Return total number of weights in the policy network."""
    w1 = INPUT_DIM * HIDDEN_DIM
    b1 = HIDDEN_DIM
    w2 = HIDDEN_DIM * OUTPUT_DIM
    b2 = OUTPUT_DIM
    return w1 + b1 + w2 + b2


def unpack_policy(policy):
    """Unpack flat weight vector into network weight matrices and biases."""
    idx = 0

    w1 = policy[idx: idx + INPUT_DIM * HIDDEN_DIM].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += INPUT_DIM * HIDDEN_DIM

    b1 = policy[idx: idx + HIDDEN_DIM]
    idx += HIDDEN_DIM

    w2 = policy[idx: idx + HIDDEN_DIM * OUTPUT_DIM].reshape(HIDDEN_DIM, OUTPUT_DIM)
    idx += HIDDEN_DIM * OUTPUT_DIM

    b2 = policy[idx: idx + OUTPUT_DIM]

    return w1, b1, w2, b2


def forward(policy, observation):
    """
    Forward pass through the two-layer neural network.

    Architecture:
        Input (8) -> Linear -> Tanh -> Hidden (16) -> Linear -> Output (4)

    Returns raw logits (pre-argmax).
    """
    w1, b1, w2, b2 = unpack_policy(policy)

    x = np.array(observation, dtype=np.float32)
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2

    return logits


def policy_action(policy, observation):
    """
    Select a discrete action given a flat policy weight vector and an observation.

    Parameters
    ----------
    policy      : 1-D numpy array of shape (get_policy_size(),)
                  Flat vector containing all network weights and biases.
    observation : array-like of length 8
                  LunarLander-v3 observation vector.

    Returns
    -------
    action : int in {0, 1, 2, 3}
    """
    logits = forward(policy, observation)
    return int(np.argmax(logits))
    # return int(np.argmax(logits + np.random.randn(4) * 0.01))