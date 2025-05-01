import numpy as np
from typing import Callable, Tuple, Dict, List
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

class Estrutura3DEvolutiva:
    def __init__(self,
                 avaliador: Callable[[np.ndarray, list], float],
                 pop_size: int = 30,
                 F: float = 0.8,
                 CR: float = 0.9,
                 geracoes: int = 100,
                 limites_dim: Dict[str, Tuple[float, float]] = None,
                 dist_minima: float = 0.05,
                 dist_maxima_conexao: float = 0.4):
        self.avaliador = avaliador
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.geracoes = geracoes
        self.dist_minima = dist_minima
        self.dist_max_conexao = dist_maxima_conexao
        self.limites_dim = limites_dim or {
            'x': (-0.3, 0.3),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.0)
        }
        self.lower_bounds = np.array([self.limites_dim['x'][0],
                                      0.0,
                                      self.limites_dim['z'][0]])
        self.upper_bounds = np.array([self.limites_dim['x'][1],
                                      self.limites_dim['y'][1],
                                      self.limites_dim['z'][1]])

    def gerar_populacao_inicial(self, max_nos: int = 100) -> List[np.ndarray]:
        pop = []
        n_nos = np.random.randint(10, max_nos // 2 + 1, size=self.pop_size)  # metade apenas

        for num in n_nos:
            coords_half = np.random.uniform(
                low=self.lower_bounds,
                high=[self.upper_bounds[0], self.upper_bounds[1] / 2, self.upper_bounds[2]],
                size=(num, 3)
            )
            coords_mirror = coords_half.copy()
            coords_mirror[:, 1] *= -1  # espelhamento em Y

            coords_total = np.vstack([coords_half, coords_mirror])
            pop.append(coords_total)
        return pop

    def gerar_conexoes(self, nos: np.ndarray) -> List[Tuple[int, int]]:
        N = len(nos)
        if N == 0:
            return []

        dist_matrix = distance_matrix(nos, nos)
        mask = (dist_matrix > self.dist_minima) & (dist_matrix < self.dist_max_conexao)
        np.fill_diagonal(mask, False)
        conexoes = list(zip(*np.where(np.triu(mask))))
        return conexoes

    def avaliar(self, individuo: np.ndarray) -> float:
        conexoes = self.gerar_conexoes(individuo)
        if len(conexoes) == 0:
            return float('inf')

        graus = np.zeros(len(individuo), dtype=int)
        if len(conexoes) > 0:
            i, j = zip(*conexoes)
            i = np.asarray(i, dtype=int)
            j = np.asarray(j, dtype=int)
            np.add.at(graus, i, 1)
            np.add.at(graus, j, 1)

        desconectados = np.sum(graus < 2)
        excessivos = max(0, len(individuo) - 100)
        penalidade = 100 * desconectados + 10 * excessivos
        return self.avaliador(individuo, conexoes) + penalidade

    def mutar(self, pop: List[np.ndarray], i: int) -> np.ndarray:
        indices = [j for j in range(self.pop_size) if j != i]
        a, b, c = np.random.choice(indices, 3, replace=False)

        # Pega a menor quantidade de nós entre os três indivíduos para garantir compatibilidade
        min_len = min(len(pop[a]), len(pop[b]), len(pop[c])) // 2  # //2 porque é simétrico

        # Pega só a parte positiva de Y (metade)
        A_half = pop[a][:min_len]
        B_half = pop[b][:min_len]
        C_half = pop[c][:min_len]

        diff = B_half - C_half
        mutant_half = A_half + self.F * diff

        # Garante simetria espelhando o eixo Y
        mutant_mirror = mutant_half.copy()
        mutant_mirror[:, 1] *= -1
        mutant = np.vstack([mutant_half, mutant_mirror])
        return mutant


    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        min_len = min(len(target), len(mutant))
        half_len = min_len // 2

        trial_half = np.copy(target[:half_len])
        mask = np.random.rand(half_len, 3) < self.CR
        trial_half[mask] = mutant[:half_len][mask]

        # Espelha para garantir simetria
        trial_mirror = trial_half.copy()
        trial_mirror[:, 1] *= -1
        trial = np.vstack([trial_half, trial_mirror])
        trial = np.clip(trial, self.lower_bounds, self.upper_bounds)
        return trial

    def evoluir(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        pop = self.gerar_populacao_inicial()
        fitness = np.array([self.avaliar(ind) for ind in pop])

        for gen in range(self.geracoes):
            print(f"Geração {gen + 1}/{self.geracoes}", end="\r")
            nova_pop = pop.copy()
            nova_fitness = fitness.copy()

            for i in range(self.pop_size):
                mutant = self.mutar(pop, i)
                trial = self.crossover(pop[i], mutant)
                trial_fitness = self.avaliar(trial)

                if trial_fitness < fitness[i]:
                    nova_pop[i] = trial
                    nova_fitness[i] = trial_fitness

            pop, fitness = nova_pop, nova_fitness

        melhor_idx = np.argmin(fitness)
        melhor = pop[melhor_idx]
        return melhor, self.gerar_conexoes(melhor)

    def plotar_estrutura(self, nos: np.ndarray, conexoes: List[Tuple[int, int]]):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(nos[:, 0], nos[:, 1], nos[:, 2], c='blue', s=20)

        if conexoes:
            linhas = np.array([[nos[i], nos[j]] for i, j in conexoes])
            ax.add_collection3d(Line3DCollection(linhas, colors='black', linewidths=0.5))

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Estrutura 3D Final")
        plt.tight_layout()
        plt.show()




def avaliador_exemplo(nos: np.ndarray, conexoes: list) -> float:
    if len(conexoes) < 10:
        return float('inf')
    comprimento_total = sum(np.linalg.norm(nos[i] - nos[j]) for i, j in conexoes)
    return comprimento_total / len(conexoes) - 0.1 * len(conexoes)

if __name__ == "__main__":
    parametros = {
        "avaliador": avaliador_exemplo,
        "pop_size": 100,
        "F": 0.7,
        "CR": 0.85,
        "geracoes": 100,
        "dist_minima": 0.05,
        "dist_maxima_conexao": 0.6
    }

    evolutiva = Estrutura3DEvolutiva(**parametros)
    melhor_individuo, conexoes = evolutiva.evoluir()

    print(f"\nMelhor indivíduo encontrado com {len(melhor_individuo)} nós e {len(conexoes)} conexões")
    evolutiva.plotar_estrutura(melhor_individuo, conexoes)
