import numpy as np

def find_suspension_nodes(nodes, sPos):
    """
    Encontra o nó mais próximo das coordenadas aproximadas da suspensão.

    Args:
        nodes (np.array): Um array NumPy de forma (n, 3) onde 'n' é o número de nós,
                                e cada linha contém [x, y, z] de um nó.
        susNode (np.array): Coordenadas aproximadas da suspensão [x, y, z]

    Returns:
        closest_index (int): Indice do nó mais proximo das coordenadas da suspensão
    """
    nodesNum = nodes.shape[0]

    closest_index = -1          # Para armazenar o índice do nó mais próximo
    closest_distance=float('inf')
    # Itera por todos os outros nós para calcular a distância
    for i in range(nodesNum):

        actual_node_coords = nodes[i]

        # Calcula a distância euclidiana entre os dois nós
        # Distância = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        distance = np.linalg.norm(sPos - actual_node_coords)

        # Se a distância atual for menor que a menor distância encontrada até agora
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i

    return closest_index

nodes = np.array([
    [0.32, 1.92, 0.0],    #0
    [0.32, 1.92, 0.48],   #1
    [0.32, 1.77, 0.21],   #2
    [0.32, 1.5, 0.03],    #3**
    [0.28, 1.14, 0.03],   #4
    [0.32, 1.14, 0.09],   #5
    [0.32, 1.23, 0.36],   #6
    [0.28, 1.14, 0.70],   #7
    [0.32, 0.63, 0.54],   #8
    [0.32, 0.69, 0.24],   #9
    [0.32, 0.69, 0.0],    #10
    [0.32, 0.45, 0.21],   #11
    [0.28, 0.24, 0.09],   #12**
    [0.16, 0.0, 0.21],    #13
    [0.16, 0.0, 0.09],    #14
    [0.16, 0.0, 0.42],    #15
    [0.28, 0.33, 0.66],   #16
    [0.28, 0.57, 1.2],    #17
    [0.0, 0.54, 1.35],    #18
    [0.00, 1.14, 0.75],   #19
    [0.00, 1.92, 0.0],    #20
    [0.00, 1.92, 0.48],   #21
    [0.00, 0.33, 0.66],   #22
    [0.00, 0.0, 0.21],    #23
    [0.00, 0.0, 0.09],    #24
    [0.00, 0.0, 0.42],    #25
])

frontSC=[0.32, 1.5, 0.03] #Coordenadas aproximadas da suspensão frontal
front_suspension_index=find_suspension_nodes(nodes,frontSC)

rearSC=[0.28, 0.24, 0.09] #Coordenadas aproximadas da suspensão traseira
rear_suspension_index=find_suspension_nodes(nodes,rearSC) #Indice do nó mais próximo da posição da suspensão

print(front_suspension_index,rear_suspension_index)