"""
Un ejemplo simple de aprendizaje por refuerzo utilizando el método Q-learning de búsqueda de tablas.
Un agente "o" está a la izquierda de un mundo unidimensional, el tesoro está en el lugar más a la derecha.
Ejecute este programa y vea cómo el agente mejorará su estrategia para encontrar el tesoro

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

np.random.seed(2)  # reproducible


N_STADOS = 10   # La longitud del mundo unidimensional.
ACTIONS = ['left', 'right']     # acciones disponibles
EPSILON = 0.9   # política codiciosa (greedy)
ALPHA = 0.1     # tasa de aprendizaje (learning rate)
GAMMA = 0.9    # factor de descuento
MAX_EPISODIOS = 15   # episodios máximos
NUEVO_TIME = 0.1    # nuevo tiempo para un movimiento
"""
Definimos constantes y parámetros que se utilizarán en el algoritmo de aprendizaje por refuerzo
"""

puntaje_acumulado_por_episodio = []
numero_de_episodios = []



def Crear_q_table(n_stados, actions):
    table = pd.DataFrame(
        np.zeros((n_stados, len(actions))),     # q_table valores iniciales
        columns=actions,    # nombre de las acciones
    )
    #print(table)    # mostrar tabla
    return table
"""
Creamos una tabla Q inicializada con valores de cero para cada estado y acción posible 
La tabla Q se almacena en un DataFrame y se devuelve como resultado de la función
"""


def eligir_action(stados, q_table):    
    stados_actions = q_table.iloc[stados, :]
    if (np.random.uniform(0,1) < EPSILON) or ((stados_actions == 0).all()): # Comprueba si se debe actuar de manera no codiciosa (explorar) o si no hay valores para las acciones.
        action_name = np.random.choice(ACTIONS) #actuar no codiciosamente (explorar)
    else:   # act greedy
        action_name = stados_actions.idxmax()   #actuar de manera codiciosa (explotar)
    return action_name
"""
Decide si el agente debe tomar una acción greedy (basada en los valores de la tabla Q) 
o de manera no greedy (explorando aleatoriamente) 
La decisión se basa en epsilon y en los valores de la tabla Q 
Si el agente elige actuar no codiciosamente o si no hay valores en la tabla Q para las acciones, 
se selecciona una acción al azar (explorar). Si elige actuar codiciosamente, selecciona la acción 
con el valor máximo en la tabla Q para el estado actual y la devuelve como resultado.
"""


def get_entorno_feedback(S, A): #S es el estado, A es la accion
    
    if A == 'right':    # mover a la derecha
        if S == N_STADOS - 2:   # agente en el penultimo estado
            S_ = 'terminal' #termina
            R = 1   #recompensa
        else:   
            S_ = S + 1  #mover a un nuevo estado
            R = 0   #sin recompensa
    else:   # mover hacia la izquierda
        R = 0   #sin recompensa
        if S == 0:  #el agente esta en la promera posicion
            S_ = S  #se queda en su lugar
        else:
            S_ = S - 1  #movimiento izquierda(nuevo estado)
    return S_, R
"""
La función calcula el nuevo estado S_ y la recompensa R que el agente recibe 
como resultado de tomar la acción
"""


def actualizar_entorn(S, episodio, pasos_contadr):
    
    #print('_______________________________')
    salto_vac = '                                      '
    entorno_list =['Inicio']+['-']*(N_STADOS-1) + ['Final']   #T es el tesoro y -
    if S == 'terminal': #Si el estado del agente es 'terminal', se muestra episodio y la cantidad total de acciones tomadas
        interaction ='Episode %s:total_pasos = %s' % (episodio+1, pasos_contadr)
       
        print('\r{}'.format(interaction), end='')
        time.sleep(2)   #espera 2 segundos
        print('\r                                ', end='') #Borra el mensaje
    else:   #Si el estado del agente no es 'terminal', se actualiza la representación
            #con la posición actual del agente
        entorno_list[S] = 'o'
        interaction = ''.join(entorno_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(NUEVO_TIME)
"""
Representación visual del entorno para mostrar el estado del agente
"""
        

def Aprendizje_Refuerzo ():
    
    q_table = Crear_q_table(N_STADOS, ACTIONS)  #inicializando tabla Q llamando a la funcion
    
    for episodio in range(MAX_EPISODIOS):   #itera de acuardo al numero de episodios establecido
        pasos_contadr = 0   #en cada episodio se inicializa el contador de pasos
        S = 0
        is_terminated = False   #
        actualizar_entorn(S, episodio, pasos_contadr)
        while not is_terminated:    #mientras no sea true

            A = eligir_action(S, q_table)   #elegir accion
            next_Stado, Reconp = get_entorno_feedback(S, A)  #obtener el siguiente estado y recompensa
            q_predict = q_table.loc[S, A]   #valor actual en la tabla
            if next_Stado != 'terminal':
                q_objetiv = Reconp + GAMMA * (q_table.iloc[next_Stado, :].max())   #todo: estima valor de Q para la acción con algoritmo de QLearning
            else:
                q_objetiv = Reconp     # recompensa
                is_terminated = True    # terminar este episodio

            q_table.loc[S, A] += ALPHA * (q_objetiv - q_predict)  #todo: actualizar con algoritmo de QLearning
            S = next_Stado  #pasar al siguiente estado

            actualizar_entorn(S, episodio, pasos_contadr+1) #actualiza el entorno
            pasos_contadr += 1  #incrementamos el contador

            EPISODIOS = episodio
            PUNTOS = pasos_contadr
            if (EPISODIOS <= MAX_EPISODIOS):
                puntaje_acumulado_por_episodio.append(PUNTOS)
                numero_de_episodios.append(EPISODIOS)
                
            # Al final del programa, crea la curva de aprendizaje
    plt.figure()
    plt.plot(numero_de_episodios, puntaje_acumulado_por_episodio)
    plt.title("Curva de Aprendizaje Q-Learning")
    plt.xlabel("Número de Episodios")
    plt.ylabel("Puntaje Acumulado")
    plt.show()

    # # Graficar la tabla Q
    # plt.figure(figsize=(8, 6))
    # plt.imshow(q_table, cmap='coolwarm')
    # plt.colorbar()
    # plt.xticks(np.arange(len(ACTIONS)), ACTIONS)
    # plt.yticks(np.arange(N_STADOS))
    # plt.xlabel('Acciones')
    # plt.ylabel('Estados')
    # plt.title('Tabla Q')
    # plt.show()
    
    return q_table
"""
Funcion principal de QLearning
"""

    


if __name__ == "__main__":
    print('\n\n\n')

    q_table = Aprendizje_Refuerzo ()
    print('\r\nQ-table:\n')
    #print(q_table)
    
    q_table_e = q_table[:-1]
    
    valores_redondeados=[round(q_table_e,3)]
    print(valores_redondeados)
    
    plt.figure()
    plt.plot(q_table_e)
    plt.title("Curva de Aprendizaje Q-Table")
    plt.xlabel("Número de Stados")
    plt.ylabel("Acciones")
    plt.show()
    plt.show()
    
    