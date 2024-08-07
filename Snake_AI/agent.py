import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Constantes para o treinamento do agente
MAX_MEMORY = 1_000_000  # Número máximo de experiências a serem armazenadas na memória
BATCH_SIZE = 10_000     # Tamanho do lote para treinamento
LR = 0.001            # Taxa de aprendizado

class Agent:
    def __init__(self):
        self.n_games = 0  # Contador de jogos jogados
        self.epsilon = 0  # Taxa de aleatoriedade para exploração
        self.gamma = 0.9  # Taxa de desconto para futuras recompensas
        self.memory = deque(maxlen=MAX_MEMORY)  # Memória para armazenar experiências
        self.model = Linear_QNet(11, 256, 3)  # Modelo de rede neural
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Treinador do modelo

    def get_state(self, game):
        head = game.snake[0]  # Posição da cabeça da cobra
        point_l = Point(head.x - 20, head.y)  # Ponto à esquerda da cabeça
        point_r = Point(head.x + 20, head.y)  # Ponto à direita da cabeça
        point_u = Point(head.x, head.y - 20)  # Ponto acima da cabeça
        point_d = Point(head.x, head.y + 20)  # Ponto abaixo da cabeça

        # Direção atual da cobra
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Estado que será passado para a rede neural
        state = [
            # Perigo à frente
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Perigo à direita
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Direção de movimento
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Localização da comida
            game.food.x < game.head.x,  # Comida à esquerda
            game.food.x > game.head.x,  # Comida à direita
            game.food.y < game.head.y,  # Comida acima
            game.food.y > game.head.y   # Comida abaixo
        ]

        return np.array(state, dtype=int)  # Retorna o estado como um array NumPy

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Adiciona a experiência na memória

    def train_long_memory(self):
        # Treina o modelo com uma amostra aleatória da memória, se houver experiências suficientes
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Amostra aleatória de experiências
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Desempacota a amostra
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Treina o modelo

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)  # Treina o modelo com uma única experiência

    def get_action(self, state):
        # Decidindo a ação com base na exploração e na exploração
        self.epsilon = 80 - self.n_games  # Diminui a taxa de aleatoriedade com o tempo
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # Escolhe uma ação aleatória
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # Converte o estado em tensor
            prediction = self.model(state0)  # Faz uma previsão usando o modelo
            move = torch.argmax(prediction).item()  # Escolhe a ação com a maior previsão
            final_move[move] = 1

        return final_move  # Retorna a ação final

def train():
    plot_scores = []  # Lista para armazenar as pontuações dos jogos
    plot_mean_scores = []  # Lista para armazenar a média das pontuações
    total_score = 0  # Pontuação total acumulada
    record = 0  # Recorde de pontuação
    agent = Agent()  # Cria uma instância do agente
    game = SnakeGameAI()  # Cria uma instância do jogo

    while True:
        # Obtém o estado antigo
        state_old = agent.get_state(game)

        # Obtém a ação a ser tomada
        final_move = agent.get_action(state_old)

        # Executa a ação e obtém o novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Treina a memória de curto prazo
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Lembra a experiência
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Treina a memória de longo prazo e plota o resultado
            game.reset()  # Reinicia o jogo
            agent.n_games += 1  # Incrementa o contador de jogos
            agent.train_long_memory()  # Treina o modelo com a memória acumulada

            if score > record:
                record = score  # Atualiza o recorde
                agent.model.save()  # Salva o modelo

            print('Jogo', agent.n_games, 'Pontuação', score, 'Recorde:', record)  # Imprime informações do jogo

            plot_scores.append(score)  # Adiciona a pontuação ao histórico
            total_score += score  # Atualiza a pontuação total
            mean_score = total_score / agent.n_games  # Calcula a média das pontuações
            plot_mean_scores.append(mean_score)  # Adiciona a média ao histórico
            plot(plot_scores, plot_mean_scores)  # Plota as pontuações

if __name__ == '__main__':
    train()
