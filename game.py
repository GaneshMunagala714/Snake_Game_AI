import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame and font for score display
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Game block size and speed
BLOCK_SIZE = 20
SPEED = 40

# Enum for directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Tuple for coordinates
Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, width=640, height=480):
        # Initialize the game window and variables
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Reset game state
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0  # to prevent agent from running forever

    def _place_food(self):
        # Place food in a random position not occupied by the snake
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # Process quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Check for collisions (game over)
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # move forward (remove tail)

        # Update the game UI and return feedback
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Check wall collision
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True
        # Check self collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Draw everything on the screen
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Action is one-hot: [straight, right turn, left turn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # keep moving straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # turn right
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # turn left

        self.direction = new_dir

        # Update head position based on direction
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
