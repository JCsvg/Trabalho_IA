import matplotlib.pyplot as plt
from IPython import display

# Ativa o modo interativo do Matplotlib, que permite a atualização contínua dos gráficos
plt.ion()

# Função para plotar os scores e a média dos scores durante o treinamento
def plot(scores, mean_scores):
    # Limpa a saída atual e exibe a figura atualizada
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()  # Limpa a figura atual para que possamos desenhar um novo gráfico
    plt.title('Treinamento...')  # Define o título do gráfico
    plt.xlabel('Numero de Jogos')  # Define o rótulo do eixo X
    plt.ylabel('Pontuação')  # Define o rótulo do eixo Y
    plt.plot(scores)  # Plota a lista de scores
    plt.plot(mean_scores)  # Plota a lista de média dos scores
    plt.ylim(ymin=0)  # Define o limite inferior do eixo Y para 0
    # Adiciona o valor do último score no gráfico
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    # Adiciona o valor da última média de scores no gráfico
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)  # Mostra o gráfico sem bloquear a execução do código
    plt.pause(.1)  # Faz uma pequena pausa para permitir que o gráfico seja atualizado
