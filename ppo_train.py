from ppo_agent import PPOAgent
from game import SnakeGameAI
import matplotlib.pyplot as plt
import os
import csv

def plot(scores, avg_rewards):
    plt.clf()
    plt.title('PPO Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score / Reward')
    plt.plot(scores, label='Score')
    plt.plot(avg_rewards, label='Avg Reward')
    plt.legend()
    plt.pause(0.1)

def train():
    env = SnakeGameAI()
    agent = PPOAgent()
    scores = []
    rewards = []
    avg_rewards = []
    total_score = 0
    prev_score = 0 

    os.makedirs("ppo_logs", exist_ok=True)
    with open("ppo_logs/ppo_training_log.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "score", "reward", "avg_reward"])

        for episode in range(1, 1001):  # Train for 1000 episodes
            state = agent.get_state(env)
            done = False
            score = 0
            episode_reward = 0

            while not done:
                action, log_prob = agent.get_action(state)
                # Store previous state for distance calc
                prev_head = env.snake[0]
                food_pos = env.food

                reward, done, score = env.play_step([1 if i == action else 0 for i in range(3)])

                # Reward shaping logic
                # Get reward based on survival and progress
                if reward == -10:
                    shaped_reward = -10  # Died
                elif score > prev_score:
                    shaped_reward = 10   # Ate food
                else:
                    shaped_reward = -0.1  # Minor penalty to discourage idle movement


            next_state = agent.get_state(env)
            agent.store_transition(state, action, log_prob, shaped_reward)
            state = next_state
            episode_reward += shaped_reward


            agent.update()

            scores.append(score)
            rewards.append(episode_reward)
            total_score += episode_reward
            avg_reward = total_score / episode
            avg_rewards.append(avg_reward)

            writer.writerow([episode, score, round(episode_reward, 2), round(avg_reward, 2)])
            print("Episode", episode, "| Score:", score, "| Reward:", round(episode_reward, 2), "| Avg Reward:", round(avg_reward, 2))
            plot(scores, avg_rewards)

if __name__ == '__main__':
    plt.ion()
    train()
