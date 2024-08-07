import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# Cores RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w  # Largura da tela
        self.h = h  # Altura da tela
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Coleta a entrada do usuário (eventos de teclado, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Se o jogador fechar a janela, encerra o jogo
                pygame.quit()
                quit()

        # 2. Move a cobra de acordo com a ação fornecida
        self._move(action)  # Atualiza a posição da cabeça da cobra
        self.snake.insert(0, self.head)  # Insere a nova posição da cabeça no início da lista da cobra

        # 3. Verifica se o jogo terminou
        reward = 0
        game_over = False

        # Determina se o jogo terminou com base nas seguintes condições
        # 1. Se a cobra colidiu com as paredes
        # 2. Se a cobra colidiu com ela mesma
        # 3. Se a cobra andou uma qunatidade de Blocos e n aconteceu nada (impedimento de loop)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score  # Retorna a recompensa, o estado de término do jogo e a pontuação atual

        # 4. Coloca nova comida ou apenas move a cobra
        if self.head == self.food:
            self.score += 1
            reward = 10  # Recompensa por comer a comida
            self._place_food()
        else:
            self.snake.pop()  # Remove o último bloco da cobra (movimento da cobra)

        # 5. Atualiza a interface gráfica e o relógio do jogo
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Retorna se o jogo terminou e a pontuação atual
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        # Verifica se há uma colisão
        if pt is None:
            pt = self.head
        # Verifica colisão com as bordas da tela
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Verifica colisão com o próprio corpo
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Pontuação: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Move a cobra de acordo com a ação escolhida
        # A ação é representada por uma lista [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Se a ação for [1, 0, 0], continua na mesma direção
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]

        # Se a ação for [0, 1, 0], vira à direita
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]

        # Se a ação for [0, 0, 1], vira à esquerda
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        # Atualiza a posição da cabeça da cobra com base na nova direção
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
