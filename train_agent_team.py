"""
train_agent_team.py
===================
Trains a two-layer neural network policy for Gymnasium LunarLander-v3
using an Evolution Strategy (ES) with antithetic sampling.

Run:
    python train_agent_team.py

Output:
    best_policy_team.npy  – flat numpy array of the best policy weights found.
"""

import numpy as np
import gymnasium as gym

from policy_team import policy_action, get_policy_size

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
POPULATION_SIZE   = 100        # Number of antithetic pairs  (actual evals = 2 * POPULATION_SIZE)
# POPULATION_SIZE   = 70        # Number of antithetic pairs  (actual evals = 2 * POPULATION_SIZE)
SIGMA             = 0.02      # Standard deviation of perturbation noise
LEARNING_RATE     = 0.03      # ES gradient step size
NUM_GENERATIONS   = 1000       # Total generations to run
EPISODES_PER_EVAL = 20         # Episodes averaged per candidate evaluation
# EPISODES_PER_EVAL = 5         # Episodes averaged per candidate evaluation
MAX_STEPS         = 1000      # Max environment steps per episode
REWARD_THRESHOLD  = 300.0     # Stop early if mean reward exceeds this
PRINT_EVERY       = 10        # Log every N generations
SEED              = 42        # Reproducibility seed
HIDDEN_DIM = 32 #hidden layer
SAVE_PATH = "best_policy_team.npy"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_policy(policy, env, n_episodes=EPISODES_PER_EVAL, max_steps=MAX_STEPS):
    """
    Roll out the policy for n_episodes and return the mean total reward.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = policy_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward
    return total_reward / n_episodes


# ---------------------------------------------------------------------------
# Evolution Strategy (antithetic sampling)
# ---------------------------------------------------------------------------

def es_train():
    rng = np.random.default_rng(SEED)
    env = gym.make("LunarLander-v3")

    policy_size = get_policy_size()

    # Initialise mean weights (small random values)
    mu = rng.standard_normal(policy_size).astype(np.float32) * 0.1

    best_reward  = -np.inf
    best_policy  = mu.copy()

    print(f"Policy size     : {policy_size} parameters")
    print(f"Population pairs: {POPULATION_SIZE}  (total evals/gen: {2 * POPULATION_SIZE})")
    print(f"Sigma           : {SIGMA}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Generations     : {NUM_GENERATIONS}")
    print("-" * 60)

    for generation in range(1, NUM_GENERATIONS + 1):

        # ---- Generate antithetic noise pairs --------------------------------
        # Shape: (POPULATION_SIZE, policy_size)
        epsilons = rng.standard_normal((POPULATION_SIZE, policy_size)).astype(np.float32)

        # ---- Evaluate every perturbed candidate -----------------------------
        rewards_pos = np.zeros(POPULATION_SIZE, dtype=np.float32)
        rewards_neg = np.zeros(POPULATION_SIZE, dtype=np.float32)

        for i in range(POPULATION_SIZE):
            theta_pos = mu + SIGMA * epsilons[i]
            theta_neg = mu - SIGMA * epsilons[i]

            rewards_pos[i] = evaluate_policy(theta_pos, env)
            rewards_neg[i] = evaluate_policy(theta_neg, env)

        # ---- Combine antithetic pairs into a single reward vector -----------
        all_rewards  = np.concatenate([rewards_pos, rewards_neg])          # (2P,)
        all_epsilons = np.concatenate([epsilons, -epsilons], axis=0)       # (2P, D)

        # ---- Fitness shaping: rank-based normalisation ----------------------
        #  Rank individuals, map ranks to [-0.5, 0.5] centred uniform scores.
        ranks = np.argsort(np.argsort(all_rewards))          # double-argsort = rank
        n     = len(ranks)
        normalised = (ranks / (n - 1)) - 0.5                 # values in [-0.5, 0.5]

        # ---- ES gradient update ---------------------------------------------
        # gradient estimate: (1 / (n * sigma)) * sum_i [ R_i * epsilon_i ]
        gradient = (all_epsilons.T @ normalised) / (n * SIGMA)
        mu = mu + LEARNING_RATE * gradient

        # ---- Track best policy ----------------------------------------------
        gen_best_idx    = int(np.argmax(all_rewards))
        gen_best_reward = all_rewards[gen_best_idx]
        gen_mean_reward = float(np.mean(all_rewards))

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            # Reconstruct the best candidate weight vector
            best_policy = mu + SIGMA * all_epsilons[gen_best_idx]
            # np.save(SAVE_PATH, best_policy)
            np.save(SAVE_PATH, mu)

        # ---- Logging --------------------------------------------------------
        if generation % PRINT_EVERY == 0 or generation == 1:
            print(
                f"Gen {generation:4d} | "
                f"mean: {gen_mean_reward:8.2f} | "
                f"best this gen: {gen_best_reward:8.2f} | "
                f"all-time best: {best_reward:8.2f}"
            )

        # ---- Early stopping -------------------------------------------------
        if gen_mean_reward >= REWARD_THRESHOLD:
            print(f"\nEarly stopping at generation {generation}: "
                  f"mean reward {gen_mean_reward:.2f} >= {REWARD_THRESHOLD}")
            break

    env.close()

    # Save the current mean as well (may outperform individual samples)
    mean_reward = evaluate_policy(mu, env=gym.make("LunarLander-v3"),
                                  n_episodes=10)
    if mean_reward > best_reward:
        best_policy = mu.copy()
        np.save(SAVE_PATH, best_policy)
        print(f"\nFinal mean policy is best (reward={mean_reward:.2f}). Saved.")
    else:
        print(f"\nTraining complete. Best reward: {best_reward:.2f}")

    print(f"Best policy saved to: {SAVE_PATH}")
    return best_policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    es_train()