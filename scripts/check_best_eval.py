import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_best_eval(logs_dir):
    eval_path = os.path.join(logs_dir, "evaluations.npz")

    if not os.path.exists(eval_path):
        print(f"No evaluations file found at {eval_path}")
        return

    data       = np.load(eval_path)
    timesteps  = data["timesteps"]
    results    = data["results"]      # episode rewards — shape (n_evals, n_episodes)
    ep_lengths = data["ep_lengths"]   # episode lengths — shape (n_evals, n_episodes)
    successes  = data["successes"]    # 0/1 success flags — shape (n_evals, n_episodes)

    # success rate per eval = mean of 0/1 flags across episodes
    success_rates = successes.mean(axis=1)
    mean_rewards  = results.mean(axis=1)
    mean_lengths  = ep_lengths.mean(axis=1)
    best_idx      = success_rates.argmax()

    print("=" * 55)
    print("FRANK — Evaluation History")
    print("=" * 55)
    print(f"\nTotal evals run   : {len(timesteps)}")
    print(f"Episodes per eval : {successes.shape[1]}")
    print(f"\nBest success rate : {success_rates[best_idx]:.1%}")
    print(f"At training step  : {timesteps[best_idx]:,}")
    print(f"Mean reward then  : {mean_rewards[best_idx]:.2f}")
    print(f"Episode length    : {mean_lengths[best_idx]:.0f} steps")

    print(f"\nLast 10 evals:")
    print(f"{'Step':>12}  {'Success':>8}  {'Reward':>10}  {'Ep Length':>10}")
    print("-" * 48)

    for i in range(max(0, len(timesteps) - 10), len(timesteps)):
        print(f"{timesteps[i]:>12,}  "
              f"{success_rates[i]:>7.1%}  "
              f"{mean_rewards[i]:>10.2f}  "
              f"{mean_lengths[i]:>10.0f}")

    print(f"\nFull progression:")
    print(f"{'Step':>12}  {'Success':>8}  {'Reward':>10}")
    print("-" * 38)

    for i in range(len(timesteps)):
        marker = "  ← BEST" if i == best_idx else ""
        print(f"{timesteps[i]:>12,}  "
              f"{success_rates[i]:>7.1%}  "
              f"{mean_rewards[i]:>10.2f}{marker}")


if __name__ == "__main__":
    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base, "logs", "reach")
    check_best_eval(logs_dir)