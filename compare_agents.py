import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
dqn_data = pd.read_csv("dqn_training_log.csv")
ppo_data = pd.read_csv("ppo_training_log.csv")

# Plot Scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(dqn_data["episode"], dqn_data["score"], label="DQN Score", alpha=0.7)
plt.plot(ppo_data["episode"], ppo_data["score"], label="PPO Score", alpha=0.7)
plt.title("Score per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

# Plot Average Rewards
plt.subplot(1, 2, 2)
plt.plot(dqn_data["episode"], dqn_data["avg_reward"], label="DQN Avg Reward", alpha=0.7)
plt.plot(ppo_data["episode"], ppo_data["avg_reward"], label="PPO Avg Reward", alpha=0.7)
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.suptitle("DQN vs PPO Comparison", fontsize=16, y=1.05)
plt.show()
