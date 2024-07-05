import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Configurações do ambiente
num_linhas = 9
num_colunas = 14
num_teste = 10000
alpha = 0.1
gamma = 0.9
epsilon = 0.1
pos_inicial_robo = (3, 1)
pos_destino = (3, 11)
pos_inicial_obstaculo = (1, 7)

# Inicializa a q-table
q_table = np.zeros((num_linhas, num_colunas, num_linhas, num_colunas, 9))

# Função para obter a nova posição no grid com base na ação do robô
def obter_nova_posicao(pos, acao):
    if acao == 0:  # Cima
        return (max(pos[0] - 1, 0), pos[1])
    elif acao == 1:  # Baixo
        return (min(pos[0] + 1, num_linhas - 1), pos[1])
    elif acao == 2:  # Esquerda
        return (pos[0], max(pos[1] - 1, 0))
    elif acao == 3:  # Direita
        return (pos[0], min(pos[1] + 1, num_colunas - 1))
    elif acao == 4:  # Noroeste
        return (max(pos[0] - 1, 0), max(pos[1] - 1, 0))
    elif acao == 5:  # Nordeste
        return (max(pos[0] - 1, 0), min(pos[1] + 1, num_colunas - 1))
    elif acao == 6:  # Sudoeste
        return (min(pos[0] + 1, num_linhas - 1), max(pos[1] - 1, 0))
    elif acao == 7:  # Sudeste
        return (min(pos[0] + 1, num_linhas - 1), min(pos[1] + 1, num_colunas - 1))
    elif acao == 8:  # Ficar parado
        return pos
    else:
        return pos

# Função responsavel pela movimentação do obstaculo
def mover_obstaculo(pos):
    velocidade = random.randint(1, 3)
    nova_pos = (pos[0] + velocidade, pos[1])
    if nova_pos[0] >= num_linhas:
        nova_pos = (0, pos[1])
    return nova_pos

# Função para selecionar a ação 
def selecionar_acao(pos_robo, pos_obstaculo):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 8)
    else:
        return np.argmax(q_table[pos_robo[0], pos_robo[1], pos_obstaculo[0], pos_obstaculo[1]])

# Função para calcular a recompensa
def calcular_recompensa(pos_robo, pos_obstaculo):
    if pos_robo == pos_destino:
        return 100
    elif pos_robo == pos_obstaculo:
        return -100
    else:
        return -1

# Função para realizar o treinamento
def treinar_q_learning(num_teste):
    global q_table
    pontuacoes = []
    for teste in range(num_teste):
        pos_robo = pos_inicial_robo
        pos_obstaculo = pos_inicial_obstaculo
        pontuacao = 0  # pontuacao inicial em cada teste

        while pos_robo != pos_destino:
            acao = selecionar_acao(pos_robo, pos_obstaculo)
            nova_pos_robo = obter_nova_posicao(pos_robo, acao)
            pos_obstaculo = mover_obstaculo(pos_obstaculo)
            recompensa = calcular_recompensa(nova_pos_robo, pos_obstaculo)
            pontuacao += recompensa  # Acumula a pontuação

            # Atualiza a q-table
            q_atual = q_table[pos_robo[0], pos_robo[1], pos_obstaculo[0], pos_obstaculo[1], acao]
            max_q_futuro = np.max(q_table[nova_pos_robo[0], nova_pos_robo[1], pos_obstaculo[0], pos_obstaculo[1]])
            q_table[pos_robo[0], pos_robo[1], pos_obstaculo[0], pos_obstaculo[1], acao] = \
                q_atual + alpha * (recompensa + gamma * max_q_futuro - q_atual)

            pos_robo = nova_pos_robo

        pontuacoes.append(pontuacao)
        print(f"Teste {teste + 1}: Pontuação {pontuacao}")
    
    return pontuacoes

# Função para testar o desempenho após o treinamento
def testar_q_learning():
    pos_robo = pos_inicial_robo
    pos_obstaculo = pos_inicial_obstaculo

    caminho_robo = [pos_robo]
    caminho_obstaculo = [pos_obstaculo]

    while pos_robo != pos_destino:
        acao = selecionar_acao(pos_robo, pos_obstaculo)
        pos_robo = obter_nova_posicao(pos_robo, acao)
        pos_obstaculo = mover_obstaculo(pos_obstaculo)

        caminho_robo.append(pos_robo)
        caminho_obstaculo.append(pos_obstaculo)

    # Visualização do caminho
    fig, ax = plt.subplots()

    # Função para atualizar o gráfico
    def update(num):
        ax.clear()
        ax.set_xlim(-1, num_colunas)
        ax.set_ylim(-1, num_linhas)

        # Desenha o robô
        robo = patches.Circle((caminho_robo[num][1], num_linhas - 1 - caminho_robo[num][0]), 0.3, color='blue')
        ax.add_patch(robo)

        # Desenha o obstáculo
        obstaculo = patches.Rectangle((caminho_obstaculo[num][1] - 0.5, num_linhas - 1 - caminho_obstaculo[num][0] - 0.5), 1, 1, color='black')
        ax.add_patch(obstaculo)

        # Desenha o destino
        ax.plot(pos_destino[1], num_linhas - 1 - pos_destino[0], 'kx', markersize=15)

    # Cria a animação
    ani = FuncAnimation(fig, update, frames=len(caminho_robo), interval=500, repeat=False)
    plt.show()

# Função principal para organizar o fluxo do programa
def main():
    print("Iniciando treinamento...")
    pontuacoes = treinar_q_learning(num_teste)
    print("Treinamento concluído. Iniciando simulação...")
    testar_q_learning()
    print("Fim das simulações.")

    # Plotar a evolução da pontuação ao longo dos testes
    plt.figure()
    plt.plot(pontuacoes)
    plt.xlabel('Teste')
    plt.ylabel('Pontuação')
    plt.title('Evolução da Pontuação ao Longo dos Testes')
    plt.show()

if __name__ == "__main__":
    main()
