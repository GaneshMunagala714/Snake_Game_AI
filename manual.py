import pygame
import random
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Constants
BLOCK_SIZE = 20
SPEED = 10
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

Point = namedtuple('Point', 'x, y')

class SnakeGameManual:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Manual Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = 'RIGHT'
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != 'RIGHT':
                    self.direction = 'LEFT'
                elif event.key == pygame.K_RIGHT and self.direction != 'LEFT':
                    self.direction = 'RIGHT'
                elif event.key == pygame.K_UP and self.direction != 'DOWN':
                    self.direction = 'UP'
                elif event.key == pygame.K_DOWN and self.direction != 'UP':
                    self.direction = 'DOWN'

        self._move()
        self.snake.insert(0, self.head)

        # Game over checks
        if self._is_collision():
            self.reset()

        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

    def _is_collision(self):
        # Wall collision
        if self.head.x >= self.w or self.head.x < 0 or self.head.y >= self.h or self.head.y < 0:
            return True
        # Self collision
        if self.head in self.snake[1:]:
            return True
        return False

    def _move(self):
        x, y = self.head.x, self.head.y
        if self.direction == 'RIGHT':
            x += BLOCK_SIZE
        elif self.direction == 'LEFT':
            x -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            y += BLOCK_SIZE
        elif self.direction == 'UP':
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGameManual()
    while True:
        game.play_step()
