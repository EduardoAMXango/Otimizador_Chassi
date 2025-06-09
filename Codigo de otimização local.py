import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import traceback
from Estrutura_Tipada import Estrutura
import matplotlib.pyplot as plt
import os
import datetime

class ChassisDEOptimizer:
    """
    Otimizador de geometria de chassi tubular usando Differential Evolution (DE).

    Atributos:
    - base_nodes: np.ndarray, coordenadas iniciais dos nós de um lado (x>=0).
    - base_connections: list of tuple, pares de índices representando arestas base.
    - mandatory_indices: índices de nós que têm deslocamento limitado a radius_mand.
    - pop_size, F, CR, max_gens: parâmetros do DE (tamanho pop., taxa de mutação, crossover, gerações).
    - radius_mand, radius_opt: raios máximos de deslocamento para nós mandatórios e opcionais.
    - tipos_tubos: lista de strings com perfis de tubo permitidos.
    """

    def __init__(
        self,
        base_nodes: np.ndarray,
        base_connections: list,
        mandatory_indices: list,
        pop_size: int = 50,
        F: float = 0.5,
        CR: float = 0.9,
        max_generations: int = 200,
        radius_mand: float = 0.05,
        radius_opt: float = 0.10,
        use_parallel: bool = True,
        fixed_tube_indices: dict = None
    ):
        """
        Inicializa o otimizador.

        Entradas:
        - base_nodes: array (n,3) de floats.
        - base_connections: lista de tuplas (i,j).
        - mandatory_indices: lista de inteiros.
        - pop_size, F, CR, max_generations: parâmetros DE.
        - radius_mand, radius_opt: floats definindo limites de deslocamento.
        - use_parallel: bool, se True permitir paralelismo. (não utilizado neste código)
        - fixed_tube_indices, lista com os indices dos elementos que so podem ser de um tipo de tubo

        Retorno:
        - Nenhum (configura atributos internos).
        """
        self.base_nodes = base_nodes.copy()
        self.n = base_nodes.shape[0]
        self.n_tubes = len(base_connections)
        self.base_connections = base_connections
        self.mandatory = set(mandatory_indices)
        self.radius_mand = radius_mand
        self.radius_opt = radius_opt

        self.tipos_tubos = ['Tubo A', 'Tubo B', 'Tubo C', 'Tubo D']
        self.fixed_tube_indices = fixed_tube_indices or {}

        # DE parameters
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_gens = max_generations

        # genotype dimension: coords + tube_vars
        self.dim_coords = 3 * self.n
        self.dim_tubes = self.n_tubes
        self.dim = self.dim_coords + self.dim_tubes

    def reflect_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """
        Espelha coordenadas de nós no plano x=0.

        Entrada:
        - nodes: array (m,3) de floats.
        Saída:
        - mirrored: array (m,3) onde mirrored[:,0] = -nodes[:,0].
        """
        mirrored = nodes.copy()
        mirrored[:, 0] *= -1
        return mirrored

    def enforce_bounds(self, coords: np.ndarray) -> np.ndarray:
        """
        Aplica limites de deslocamento e arredonda as coordenadas.

        Entrada:
        - coords: array (n,3) de floats (proposta de deslocamento).
        Saída:
        - adjusted: array (n,3) de floats, cada deslocamento limitado a radius_mand ou radius_opt
          e arredondado para 3 casas decimais.
        """
        adjusted = coords.copy()
        for i in range(self.n):
            orig = self.base_nodes[i]
            delta = coords[i] - orig
            dist = np.linalg.norm(delta)
            r_max = self.radius_mand if i in self.mandatory else self.radius_opt
            if dist > r_max:
                adjusted[i] = orig + delta / dist * r_max
        # Arredonda para 3 casas decimais para estabilidade da evolução
        adjusted = np.round(adjusted, 3)
        return adjusted

    def decode_individual(self, x: np.ndarray):
        """
        Converte um vetor genotípico em nós completos e lista de elementos.

        Entrada:
        - x: array (dim_coords+dim_tubes,) de floats.
        Saída:
        - nodes_full: array (N,3) de floats com nós de ambos os lados.
        - elements: lista de tuplas (i, j, perfil), com índices nos nodes_full.

        Processos:
        1. Extrai coords e tube_vars do vetor x.
        2. Aplica enforce_bounds às coords.
        3. Para nós com base x≈0, inclui apenas um nó (central).
           Para outros, inclui coord e seu espelho.
        4. Usa mapeamento para gerar conexões simétricas com tipo de tubo.
        """
        # Separa coords e tube_vars
        coords = x[:self.dim_coords].reshape((self.n, 3))
        tube_vars = x[self.dim_coords:]

        # Aplica bounds
        coords = self.enforce_bounds(coords)

        # Monta os nós completos com mapeamento para índices
        full_nodes = []
        mapping = {}
        for i, coord in enumerate(coords):
            # central: x original == 0
            if np.isclose(self.base_nodes[i, 0], 0.0):
                idx = len(full_nodes)
                full_nodes.append(coord)
                mapping[i] = [idx]
            else:
                # lateral: coordenada e seu espelho
                idx1 = len(full_nodes)
                full_nodes.append(coord)
                mirrored = coord.copy()
                mirrored[0] *= -1
                idx2 = len(full_nodes)
                full_nodes.append(mirrored)
                mapping[i] = [idx1, idx2]

        nodes_full = np.array(full_nodes)

        # Monta conexões com tipo de tubo simétrico
        elements = []
        for idx_conn, (i, j) in enumerate(self.base_connections):
            # determina perfil a partir de tube_vars
            if idx_conn in self.fixed_tube_indices:
                perfil = self.fixed_tube_indices[idx_conn]
            else:
                tval = tube_vars[idx_conn]
                t_int = int(np.clip(np.floor(tval), 0, len(self.tipos_tubos) - 1))
                perfil = self.tipos_tubos[t_int]

            # combina índices conforme mapeamento
            ids_i = mapping[i]
            ids_j = mapping[j]
            # se ambos centrais
            if len(ids_i) == 1 and len(ids_j) == 1:
                elements.append((ids_i[0], ids_j[0], perfil))
            # se ambos laterais, conecta pares correspondentes
            elif len(ids_i) == 2 and len(ids_j) == 2:
                elements.append((ids_i[0], ids_j[0], perfil))
                elements.append((ids_i[1], ids_j[1], perfil))
            # se um central e outro lateral, conecta central a ambos
            else:
                # identifica qual é central
                if len(ids_i) == 1:
                    cent = ids_i[0]; lats = ids_j
                else:
                    cent = ids_j[0]; lats = ids_i
                for lat in lats:
                    elements.append((cent, lat, perfil))

        return nodes_full, elements

    def validate_min_distance(self, coords, min_dist=0.05):
        """
        Verifica se todas as distâncias entre pares de nós >= min_dist.

        Entrada:
        - coords: array (M,3).
        - min_dist: float.
        Saída:
        - bool, True se válido.
        """
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return np.all(d >= min_dist)

    def initialize_individual(self) -> np.ndarray:
        """
        Gera um indivíduo válido.

        Saída:
        - x: array (dim,) de floats, contendo coords arredondadas e tube_vars iniciais.
        """
        while True:
            # Gera coords aleatórias
            deltas = np.random.normal(size=(self.n, 3))
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            deltas = deltas / norms * (np.random.rand(self.n,1) ** (1/3) * self.radius_opt)
            coords = self.base_nodes + deltas
            coords = self.enforce_bounds(coords)
            # tube_vars iniciais aleatórios entre [0,4)
            tube_vars = np.random.uniform(0, len(self.tipos_tubos), size=(self.n_tubes,))
            for idx, tipo in self.fixed_tube_indices.items():
                t_idx = self.tipos_tubos.index(tipo)
                tube_vars[idx] = t_idx + 0.01  # pequeno offset para não arredondar para o próximo

            x = np.concatenate([coords.reshape(-1), tube_vars])
            nodes, _ = self.decode_individual(x)

            if self.validate_min_distance(nodes):
                #print("Individuo valido" ,end="\r")
                return x

    def initialize_population(self):
        """
        Constrói a população inicial.

        Saída:
        - pop: array (pop_size, dim) de indivíduos.
        - fitness: array (pop_size,) de floats.
        """
        pop = np.zeros((self.pop_size, self.dim))
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            indiv = self.initialize_individual()
            pop[i] = indiv
            fitness[i] = self.evaluate(indiv)
        return pop, fitness

    def evaluate(self, x: np.ndarray) -> float:
        """
        Avalia o custo de um indivíduo.

        Entrada:
        - x: array (dim,) de floats.
        Saída:
        - float, valor de penalidade (fitness).

        Processo:
        1. Decodifica para nodes e elements.
        2. Checa distância mínima.
        3. Monta e analisa pela classe Estrutura (FEA).
        4. Calcula penalidade via penalidade_chassi.
        """
        try:
            nodes, elements = self.decode_individual(x)
            if not self.validate_min_distance(nodes):
                return float('inf')
            estrutura = Estrutura(elements, nodes)
            estrutura.matrizes_global()
            fixed = list(range(6))
            Fg = np.zeros(estrutura.num_dofs)
            Fg[0] = 300
            Fg[2] = 300
            disp = estrutura.static_analysis(Fg, fixed)
            stresses = estrutura.compute_stress(estrutura.compute_strain(disp),210e9,0.27)
            von = estrutura.compute_von_mises(stresses)
            massa = estrutura.mass()
            *_, KT, KF, _, _ = estrutura.shape_fun()
            return penalidade_chassi(KT, KF, massa, von)
        except Exception:
            traceback.print_exc()
            return float('inf')

    def mutate_and_crossover(self, idx, pop):
        """
        Gera um novo indivíduo por DE/rand/1 + crossover binomial.

        Entradas:
        - idx: índice do indivíduo-alvo.
        - pop: array (pop_size, dim) população atual.
        Saída:
        - u: array (dim,) candidato filho (ou x_i se inválido).
        """
        idxs = list(range(self.pop_size)); idxs.remove(idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        x_i = pop[idx]
        x_a, x_b, x_c = pop[a], pop[b], pop[c]
        # mutação DE/rand/1
        v = x_a + self.F * (x_b - x_c)
        # crossover binomial
        j_rand = np.random.randint(self.dim)
        u = np.array([v[j] if np.random.rand()<self.CR or j==j_rand else x_i[j]
                      for j in range(self.dim)])
        # aplica bounds às coords
        coords = self.enforce_bounds(u[:self.dim_coords].reshape(self.n,3)).reshape(-1)
        tube_vars = np.clip(u[self.dim_coords:], 0, len(self.tipos_tubos)-1e-3)
        for idx, tipo in self.fixed_tube_indices.items():
            t_idx = self.tipos_tubos.index(tipo)
            tube_vars[idx] = t_idx + 0.01
        u = np.concatenate([coords, tube_vars])
        # valida distância
        if self.validate_min_distance(self.decode_individual(u)[0]):
            return u
        return x_i

    def optimize(self):
        """
        Executa o loop principal de otimização.

        Saída:
        - best_solution: tupla (nodes, elements) do melhor indivíduo.
        - best_cost: float, custo associado.
        - history: dict com listas de métricas por geração.
        """
        print("Otimização Iniciada")
        pop, fit = self.initialize_population()
        
        # Variáveis para controle de convergência
        convergence_count = 0
        convergence_threshold = 3  # Número de gerações consecutivas para convergência
        history = {
            'best_fit': [],
            'avg_fit': [],
            'std_dev': []
        }
        
        for gen in range(self.max_gens):
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            # Processa cada indivíduo
            for i in range(self.pop_size):
                u = self.mutate_and_crossover(i, pop)
                f_u = self.evaluate(u)
                if f_u <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = f_u
            
            pop = new_pop
            fit = new_fit
            
            # Calcula estatísticas da população
            best_fit = np.min(fit)
            avg_fit = np.mean(fit)
            std_dev = np.mean(np.std(pop, axis=0))  # Desvio padrão médio
            
            # Armazena histórico
            history['best_fit'].append(best_fit)
            history['avg_fit'].append(avg_fit)
            history['std_dev'].append(std_dev)
            
            # Verifica convergência
            if std_dev < 0.02:
                convergence_count += 1
                print(f"Gen {gen+1}/{self.max_gens} Convergindo... (Std={std_dev:.4f}, Count={convergence_count})", end='\r')
            else:
                convergence_count = 0
                print(f"Gen {gen+1}/{self.max_gens} Best={best_fit:.4e} Std={std_dev:.4f}", end='\r')
            
            # Critério de parada por convergência
            if convergence_count >= convergence_threshold:
                print(f"\nConvergência alcançada na geração {gen+1} com desvio padrão {std_dev:.4f}")
                break
        
        best_idx = np.argmin(fit)
        best_solution = self.decode_individual(pop[best_idx])
        return best_solution, fit[best_idx], history

    def plotar(self, individuo):
        """
        Plota a geometria do chassi evoluído em 3D.

        Entrada:
        - individuo: tupla (nodes, elements).
        Saída:
        - exibe plot Matplotlib.
        """
        nodes, elements = individuo
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        num_nodes = len(nodes)
        elements_valid = [(i, j, t) for i, j, t in elements if 0 <= i < num_nodes and 0 <= j < num_nodes]

        xs, ys, zs = zip(*nodes)
        ax.scatter(ys, xs, zs, s=25, c='black')
        for i, j, t in elements_valid:
            ni, nj = nodes[i], nodes[j]
            ax.plot([ni[1], nj[1]], [ni[0], nj[0]], [ni[2], nj[2]], 'b-')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.set_box_aspect([3,1,2])
        plt.title("Chassi Evoluído")
        plt.show()

    def plot_convergence(self, history, save_path=None, show=True):
        """
        Gera gráfico de convergência: fitness vs geração e std_dev.

        Entradas:
        - history: dict com 'best_fit', 'avg_fit', 'std_dev'.
        - save_path: caminho opcional para salvar o PNG.
        - show: bool, se True exibe o plot.
        """
        plt.figure(figsize=(10, 6))
        
        # Gráfico duplo eixo Y
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Curva de fitness
        ax1.plot(history['best_fit'], 'b-', linewidth=2, label='Melhor Fitness')
        ax1.plot(history['avg_fit'], 'b--', alpha=0.7, label='Fitness Médio')
        ax1.set_ylabel('Fitness', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_yscale('log')
        
        # Curva de desvio padrão
        ax2.plot(history['std_dev'], 'r-', linewidth=2, label='Desvio Padrão')
        ax2.axhline(y=0.2, color='g', linestyle='--', label='Limite Convergência')
        ax2.set_ylabel('Desvio Padrão', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Progresso da Otimização')
        plt.xlabel('Geração')
        plt.grid(True)
        
        # Unificar legendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico de convergência salvo em: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def save_solution(self, nodes, elements, file_path):
        """
        Salva nós e elementos em arquivo texto.

        Entradas:
        - nodes: array (N,3).
        - elements: lista de tuplas.
        - file_path: str, caminho de saída.
        """
        with open(file_path, 'w') as f:
            f.write("SOLUÇÃO FINAL DO CHASSI\n")
            f.write("="*50 + "\n\n")
            
            # Salva nós
            f.write("NÓS:\n")
            f.write("Índice | Coordenada X | Coordenada Y | Coordenada Z\n")
            f.write("-"*50 + "\n")
            for i, node in enumerate(nodes):
                f.write(f"{i:6d} | {node[0]:11.3f} | {node[1]:11.3f} | {node[2]:11.3f}\n")
            
            # Salva elementos
            f.write("\n\nELEMENTOS (CONEXÕES):\n")
            f.write("Índice | Nó Inicial | Nó Final | Tipo de Tubo\n")
            f.write("-"*50 + "\n")
            for i, elem in enumerate(elements):
                f.write(f"{i:6d} | {elem[0]:10d} | {elem[1]:9d} | {elem[2]}\n")
        
        print(f"Solução salva em: {file_path}")

    def print_solution(self, nodes, elements):
        """
        Imprime no console nós e elementos formatados.

        Entradas:
        - nodes: array (N,3).
        - elements: lista de tuplas.
        """
        print("\n" + "="*50)
        print("SOLUÇÃO FINAL DO CHASSI")
        print("="*50)
        
        # Imprime nós
        print("\nNÓS:")
        print("Índice | Coordenada X | Coordenada Y | Coordenada Z")
        print("-"*50)
        for i, node in enumerate(nodes):
            print(f"{i:6d} | {node[0]:11.6f} | {node[1]:11.6f} | {node[2]:11.6f}")
        
        # Imprime elementos
        print("\n\nELEMENTOS (CONEXÕES):")
        print("Índice | Nó Inicial | Nó Final | Tipo de Tubo")
        print("-"*50)
        for i, elem in enumerate(elements):
            print(f"{i:6d} | {elem[0]:10d} | {elem[1]:9d} | {elem[2]}")   

def penalidade_chassi(KT, KF, massa, tensoes):
    """
    Calcula penalidade total do chassi.

    Entradas:
    - KT, KF: floats de rigidezes.
    - massa: float.
    - tensoes: array de floats.
    Saída:
    - penalidade_total: float.
    """
    # Limites e parâmetros
    KT_min = 1e7          # Rigidez torcional mínima (N·m/rad)
    KF_min = 1e6          # Rigidez flexão mínima (N/m)
    massa_ideal = 23       # Massa alvo (kg)
    K_mola = 5e5           # Constante da mola do amortecedor (N/m)
    tensao_adm = 250e6     # Tensão admissível do material (Pa)
    alpha = 0.5            # Fator de sensibilidade exponencial
    beta = 10              # Fator de escala logarítmica
    
    penalidade_total = 0

    # 1. Rigidez Torcional (Função Exponencial)
    if KT < KT_min:
        deficit = (KT_min - KT) / KT_min
        # Penalidade cresce exponencialmente com o déficit
        penalidade_total += np.exp(alpha * deficit) - 1

    # 2. Rigidez em Flexão (Função Logarítmica)
    if KF < KF_min:
        deficit = (KF_min - KF) / KF_min
        # Penalidade logarítmica: suave para pequenas violações, forte para grandes
        penalidade_total += beta * np.log(1 + deficit)

    # 3. Massa (Função Híbrida)
    if massa > massa_ideal:
        excesso = (massa - massa_ideal) / massa_ideal
        # Combina resposta linear inicial com crescimento exponencial
        penalidade_total += excesso + np.exp(alpha * excesso) - 1

    # 4. Compatibilidade com Mola (Lógica Aprimorada)
    ratio_KT = K_mola / KT if KT > 0 else float('inf')
    ratio_KF = K_mola / KF if KF > 0 else float('inf')
    
    if ratio_KT > 25 or ratio_KF > 25:
        # Penalidade proporcional ao nível de incompatibilidade
        violacao = max(ratio_KT/25, ratio_KF/25) - 1
        penalidade_total += 100 * violacao**2

    # 5. Tensões (Abordagem Baseada em Risco)
    tensao_max = max(tensoes)
    if tensao_max > tensao_adm:
        # Penalidade exponencial para tensões acima do admissível
        excesso = (tensao_max - tensao_adm) / tensao_adm
        penalidade_total += np.exp(5 * excesso) - 1
    
    # Penalidade por distribuição desigual de tensões (logarítmica)
    razao_tensoes = np.ptp(tensoes) / np.mean(tensoes) if np.mean(tensoes) > 0 else 0
    penalidade_total += np.log(1 + razao_tensoes)

    return penalidade_total * 100  # Fator de escala global

if __name__ == "__main__":
    nodes = np.array([
    [0.32, 1.92, 0.0],    #0
    [0.32, 1.92, 0.48],   #1
    [0.32, 1.77, 0.21],   #2
    [0.32, 1.5, 0.03],    #3
    [0.28, 1.14, 0.03],   #4
    [0.32, 1.14, 0.09],   #5
    [0.32, 1.23, 0.36],   #6
    [0.28, 1.14, 0.72],   #7
    [0.32, 0.63, 0.54],   #8
    [0.32, 0.69, 0.24],   #9
    [0.32, 0.69, 0.0],    #10
    [0.32, 0.45, 0.21],   #11
    [0.28, 0.24, 0.09],   #12
    [0.16, 0.0, 0.21],    #13
    [0.16, 0.0, 0.09],    #14
    [0.16, 0.0, 0.42],    #15
    [0.28, 0.33, 0.66],   #16
    [0.28, 0.57, 1.2],    #17
    [0.0, 0.54, 1.35],    #18
    [0.0, 1.14, 0.78],     #19
    [0.00, 1.92, 0.0],    #20
    [0.00, 1.92, 0.48],   #21
    [0.00, 0.33, 0.66],   #22
])

    connections = [
    (0, 1), (0,2),(0,3),(1,7),(0,20),
    (1,6), (1,2),(0,3),(8,17),(3,6),
    (3,4), (4,5),(4,10),(1,21),(16,22),
    (2, 6), (2,3),(0,3),(5,6),(6,9),(6,10),
    (3,5),(6,7),(7,8),(5,10),(10,12),
    (8,9),(8,6),(9,10),(11,13),
    (10,11),(11,12),(11,9),
    (13, 14), (12,14),(13,15),
    (7,19),(15,16),(16,17),(8,16),
    (17,18),(11,16),(8,11),(11,15)
]

    fixed_tube={0: "Tubo B",1: "Tubo B",2: "Tubo B"}

    indices = [0,1,3,5,6,7,8,10,14,15,16,17,18,19,20,21,22]
        # Criar diretório para resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"resultados_otimizacao_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    otimizador = ChassisDEOptimizer(
        base_nodes= nodes,
        base_connections = connections,
        mandatory_indices = indices,
        pop_size=10,
        F=0.6,
        CR=0.8,
        max_generations=5,
        fixed_tube_indices=fixed_tube)

    best_indiv, best_cost, history = otimizador.optimize()
    print(f"\nMelhor custo: {best_cost:.6e}")
    nodes_final, elements_final = best_indiv
    
    # 1. Imprimir solução no console
    otimizador.print_solution(nodes_final, elements_final)
    
    # 2. Salvar solução em arquivo TXT
    solution_path = os.path.join(results_dir, "solucao_final.txt")
    otimizador.save_solution(nodes_final, elements_final, solution_path)
    
    # 3. Salvar gráfico de histórico
    convergence_path = os.path.join(results_dir, "historico_convergencia.png")
    otimizador.plot_convergence(history, save_path=convergence_path, show=True)
    
    # 4. Plotar solução final
    otimizador.plotar(best_indiv)
    
    # Salvar também a visualização 3D
    plot_path = os.path.join(results_dir, "visualizacao_3d.png")
    plt.savefig(plot_path)
    print(f"Visualização 3D salva em: {plot_path}")
