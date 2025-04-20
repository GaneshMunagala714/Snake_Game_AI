from agent import Agent
from game import SnakeGameAI
import matplotlib.pyplot as plt

# Plot score progression
def plot(scores, mean_scores):
    plt.clf()  # Clear previous plot
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()
    plt.pause(0.1)  # Pause to allow real-time updates

# Main training loop
def train():
    scores = []          # All game scores
    mean_scores = []     # Running average
    total_score = 0
    record = 0           # Highest score seen so far

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state from environment
        state_old = agent.get_state(game)

        # Get action from agent (policy or random)
        final_move = agent.get_action(state_old)

        # Perform action and get reward + game state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory on latest step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Game over — reset game
            game.reset()
            agent.n_games += 1

            # Train on experience replay
            agent.train_long_memory()

            # Save best model so far
            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score: {score} | Record: {record}')

            # Plotting
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

# Run training
if __name__ == '__main__':
    plt.ion()  # Interactive plotting on
    train()
