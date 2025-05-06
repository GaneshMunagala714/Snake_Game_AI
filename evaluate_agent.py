import torch
import pygame
import csv
from model import Linear_QNet
from game import SnakeGameAI

MODEL_PATH = 'model.pth'
CSV_PATH = 'evaluation_scores.csv'

def evaluate():
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    game = SnakeGameAI()
    total_score = 0
    num_games = 20
    all_scores = []

    for game_idx in range(num_games):
        game.reset()
        done = False
        while not done:
            state_old = game.get_state()
            prediction = model(torch.tensor(state_old, dtype=torch.float))
            final_move = [0, 0, 0]
            final_move[torch.argmax(prediction).item()] = 1

            reward, done, score = game.play_step(final_move)

        print(f"Game {game_idx + 1}: Score = {score}")
        all_scores.append(score)
        total_score += score

    average_score = total_score / num_games
    print(f"\nAverage score over {num_games} games: {average_score}")

    # Save to CSV
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Game', 'Score'])
        for i, score in enumerate(all_scores, 1):
            writer.writerow([i, score])
        writer.writerow(['Average', average_score])

if __name__ == "__main__":
    evaluate()
