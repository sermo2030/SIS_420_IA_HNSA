import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def initialize_q_table(observation_space, action_space):
    return np.zeros((observation_space, action_space))  #todo:creacion tabla Q

def choose_action(q_table, state, epsilon, is_training, action_space, rng):
    if is_training and rng.random() < epsilon:  #todo:exploracion
        return np.random.choice(action_space)
    else:
        return np.argmax(q_table[state, :]) #todo:explotacion

def update_q_table(q_table, state, action, reward, new_state, learning_rate, discount_factor):
    q_table[state, action] = q_table[state, action] + learning_rate * (
        reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]   #todo: funcion Q
    )

def decay_epsilon(epsilon, epsilon_decay_rate):
    return max(epsilon - epsilon_decay_rate, 0)

def run_q_learning(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    observation_space = env.observation_space.n
    action_space = env.action_space.n

    q_table = initialize_q_table(observation_space, action_space)

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    if not is_training:
        try:
            with open('gym_solution8x8.pkl', 'rb') as f:
                q_table = pickle.load(f)
        except FileNotFoundError:
            print("Archivo .pkl no encontrado. Asegúrate de que el archivo exista y tenga el nombre correcto.")


    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = choose_action(q_table, state, epsilon, is_training, action_space, rng) #exploracion vs explotacion

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                update_q_table(q_table, state, action, reward, new_state, learning_rate, discount_factor)   #actualizar tabla

            state = new_state

        epsilon = decay_epsilon(epsilon, epsilon_decay_rate)    #reducir epcilon

        if epsilon == 0:
            learning_rate = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('gym_solution8x8.png')

    if is_training:
        save_q_table(q_table, "gym_solution8x8.pkl")

def save_q_table(q_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

if __name__ == '__main__':
    run_q_learning(1000, is_training=True, render=True)
