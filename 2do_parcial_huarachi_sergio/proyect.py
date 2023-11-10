import numpy as np
import matplotlib.pyplot as plt

# Parámetros del Q-learning
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 0.1  # Probabilidad de exploración
maxEpi = 50

# Estados posibles (0: rojo, 1: verde)
states = [0, 1]

# Acciones posibles (0: mantener, 1: cambiar)
actions = [0, 1]

# Inicialización de la tabla Q
Q = np.zeros((len(states), len(actions)))

# Función para seleccionar una acción epsilon-greedy
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state, :])

# Función para actualizar la tabla Q
def update_Q(state, action, reward, next_state):
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))

# Función para visualizar la tabla Q
def visualize_Q(Q):
    plt.imshow(Q, cmap='Blues', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(actions)), ['Keep', 'Change'])
    plt.yticks(range(len(states)), ['Red', 'Green'])
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.title('Q-table')
    plt.show()

# Simulación del semáforo
def simulate_traffic_light(episodes):
    for episode in range(episodes):
        state = np.random.choice(states)  # Estado inicial aleatorio
        total_reward = 0

        while True:
            action = select_action(state)

            # Simulación de las consecuencias de la acción
            if state == 0 and action == 1:  # Cambiar de rojo a verde
                reward = 1
                next_state = 1
            elif state == 1 and action == 0:  # Mantener verde
                reward = 0.5
                next_state = 1
            else:
                reward = 0
                next_state = state

            total_reward += reward

            # Actualizar la tabla Q
            update_Q(state, action, reward, next_state)

            # Visualizar la tabla Q en cada paso
            visualize_Q(Q)

            print(f"Episode: {episode + 1}, State: {state}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

            if state == 1:  # Si el semáforo está en verde, terminar el episodio
                break

            state = next_state

# Simulación de 10 episodios
simulate_traffic_light(maxEpi)
