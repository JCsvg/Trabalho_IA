import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# Define uma classe para a rede neural linear usada no Q-learning
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define a primeira camada linear (total de entrada -> total da camada oculta)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Define a segunda camada linear (total da camada oculta -> total de saída)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Define a passagem para frente (forward pass)
    def forward(self, x):
        # Aplica a primeira camada e uma função de ativação ReLU
        x = F.relu(self.linear1(x))
        # Aplica a segunda camada linear
        x = self.linear2(x)
        # Retorna a saída final
        return x

    # Método para salvar o modelo em um arquivo
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        # Cria o diretório 'model' se ele não existir
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # Salva os parâmetros do modelo no arquivo especificado
        torch.save(self.state_dict(), file_name)


# Define uma classe para treinar o modelo usando Q-learning
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.model = model  # Modelo (rede neural)
        # Define o otimizador Adam com a taxa de aprendizado
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Define a função de perda como erro quadrático médio (MSE)
        self.criterion = nn.MSELoss()

    # Método para executar um passo de treinamento
    def train_step(self, state, action, reward, next_state, done):
        # Converte os estados, ações e recompensas em tensores PyTorch com tipo float
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)  # Ação é convertida para tensor de inteiros longos
        reward = torch.tensor(reward, dtype=torch.float)

        # Verifica se o estado tem apenas uma dimensão
        if len(state.shape) == 1:
            # Adiciona uma dimensão extra para representar o batch size
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Calcula os valores Q previstos para o estado atual
        pred = self.model(state)

        # Cria uma cópia dos valores Q previstos
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            # Se o episódio não terminou, atualiza o valor Q
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Atualiza o valor Q para a ação realizada
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Calcula a perda e realiza backpropagation para ajustar os pesos
        self.optimizer.zero_grad()  # Zera os gradientes acumulados
        loss = self.criterion(target, pred)  # Calcula a perda (erro)
        loss.backward()  # Realiza o cálculo dos gradientes
        self.optimizer.step()  # Atualiza os pesos do modelo com base nos gradientes
