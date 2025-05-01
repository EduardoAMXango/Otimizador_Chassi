import random
from collections import defaultdict
import numpy as np
from estrutura import Estrutura

class ChassiDE:
    def __init__(
        self,
        n_nodes=40,
        pop_size=15,
        ngen=30,
        F=0.8,
        CR=0.7,
        min_dist=0.15,
        max_dist=0.7,
        allowable_stress=250e6,
        min_freq=20,
        early_stop=None
    ):
        # Parâmetros DE
        self.subestrutura_obrigatoria = {
            'nodes': [[0.0, -0.3, 1.200],
                    [0., 0.300, 1.200],
                    [0., 0., 1.350]] 
            ,
            'elements':[(1,3),(2,3)]      }   
        self.N_NODES = n_nodes
        self.POP_SIZE = pop_size
        self.NGEN = ngen
        self.F = F
        self.CR = CR
        self.MIN_DIST = min_dist
        self.MAX_DIST = max_dist
        self.ALLOWABLE_STRESS = allowable_stress
        self.MIN_FREQ = min_freq
        # Parametros Ideais
        self.KT_I = 1e8
        self.KF_I = 1e8
        self.m_I  = 23
        # Limites espaciais
        self.MAX_X, self.MAX_Y, self.MAX_Z = 0.6, 2.0, 1.0
        # População inicial
        self.population = [self.criar_individuo_simetrico() for _ in range(pop_size)]
        # Subestrutura obrigatória (ex: triângulo na dianteira)

    def garantir_conectividade(self, nodes, elements):

        # Mapeia número de conexões por nó
        conexoes = defaultdict(int)
        for i, j in elements:
            conexoes[i] += 1
            conexoes[j] += 1

        # Para cada nó com menos de 2 conexões
        for i in range(len(nodes)):
            while conexoes[i] < 2:
                # Encontra vizinhos válidos (respeitando distância)
                distancias = []
                for j in range(len(nodes)):
                    if i != j and (i, j) not in elements and (j, i) not in elements:
                        d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
                        if self.MIN_DIST <= d <= self.MAX_DIST:
                            distancias.append((d, j))

                if not distancias:
                    break  # Sem vizinhos válidos

                distancias.sort()
                for _, j in distancias:
                    if conexoes[j] < 4:  # Limite máximo de conexões por nó (ajustável)
                        elements.append((i, j))
                        conexoes[i] += 1
                        conexoes[j] += 1
                        break  # Adicionou uma, volta para checar se ainda precisa de outra

        return elements

    def criar_individuo_simetrico(self):
            # Adiciona os nós obrigatórios
            nodes = [np.array(n) for n in self.subestrutura_obrigatoria['nodes']]
            elements = list(self.subestrutura_obrigatoria['elements'])

            half = (self.N_NODES - len(nodes)) // 2
            start_idx = len(nodes)
            
            # Cria nós simétricos em relação ao plano YZ
            for i in range(half):
                x = round(random.uniform(0.1, self.MAX_X), 3)  # Coordenada x aleatória
                y = round(random.uniform(0.0, self.MAX_Y), 3)  # Coordenada y aleatória
                z = round(random.uniform(-0.3, self.MAX_Z), 3)  # Coordenada z aleatória

                nodes.append([x, y, z])  # Adiciona o nó
                nodes.append([-x, y, z])  # Adiciona o nó simétrico

            # Ajustando a criação dos elementos
            for i in range(start_idx, len(nodes), 2):
                elements.append((i, i + 1))  # Conecta nós simétricos
            for i in range(start_idx, len(nodes) - 4, 2):
                # Conecta nós com nós a 4 posições de distância na lista
                elements.append((i, i + 4 if i + 4 < len(nodes) else start_idx + (i + 4 - len(nodes)) % (len(nodes) - start_idx)))
                elements.append((i + 1, i + 5 if i + 5 < len(nodes) else start_idx + (i + 5 - len(nodes)) % (len(nodes) - start_idx)))

            if half >= 4:
                for i in range(start_idx, start_idx + 2 * (half - 2), 2):
                    # Conecta nós com nós a 6 posições de distância na lista
                    elements.append((i, i + 6 if i + 6 < len(nodes) else start_idx + (i + 6 - len(nodes)) % (len(nodes) - start_idx)))
                    elements.append((i + 1, i + 7 if i + 7 < len(nodes) else start_idx + (i + 7 - len(nodes)) % (len(nodes) - start_idx)))
                    
            elements = self.garantir_conectividade(nodes, elements)

            return [nodes, elements]
    
    def validar_coordenadas(self,coordenadas):
        for x, y, z in coordenadas:
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x) or np.isinf(y) or np.isinf(z):
                raise ValueError(f"Coordenadas inválidas detectadas: ({x}, {y}, {z})")

    def mutacao(self, base, a, b, best):
        mutant_nodes = (
            np.array(best[0]) +
            self.F * (np.array(a[0]) - np.array(b[0]))
        )
        new_nodes = []
        
        # Define a precisão desejada (em milímetros, ou 3 casas decimais)
        precisao = 3

        for i in range(0, len(mutant_nodes), 2):
            x, y, z = mutant_nodes[i]

            # Aplicar limites nas coordenadas (mínimos e máximos)
            x = np.clip(x, 0.1, self.MAX_X)
            y = np.clip(y, 0.0, self.MAX_Y)
            z = np.clip(z, -0.3, self.MAX_Z)

            # Arredondar as coordenadas para a precisão desejada
            x = round(x, precisao)
            y = round(y, precisao)
            z = round(z, precisao)

            new_nodes.append([x, y, z])
            new_nodes.append([-x, y, z])

        # Validação das coordenadas geradas
        try:
            self.validar_coordenadas(new_nodes)  # Validar as coordenadas dos nós gerados
        except ValueError as e:
            print(f"Erro na mutação: {e}")
            return base  # Retorna o indivíduo base caso haja erro de coordenadas

        new_elements = base[1].copy()
        if random.random() < 0.2:
            i = random.randint(0, len(new_nodes) // 2 - 1) * 2
            j = random.randint(0, len(new_nodes) // 2 - 1) * 2
            dist = np.linalg.norm(
                np.array(new_nodes[i]) - np.array(new_nodes[j])
            )
            if self.MIN_DIST <= dist <= self.MAX_DIST:
                new_elements.extend([(i, j), (i + 1, j + 1)])
            if len(new_elements) > 10:
                idx = random.randint(0, len(new_elements) - 1)
                new_elements.pop(idx)

            # Reinsere elementos obrigatórios se ausentes
        for e in self.subestrutura_obrigatoria['elements']:
              if e not in new_elements and (e[1], e[0]) not in new_elements:
                  new_elements.append(e)

          # Mantém nós obrigatórios fixos (ou suavemente variáveis, se desejar)
        for i, ref_node in enumerate(self.subestrutura_obrigatoria['nodes']):
              new_nodes[i] = [round(c, 3) for c in ref_node]

        return [new_nodes, new_elements]

    def recombinacao(self, target, mutant):
        trial_nodes = []
        for t, m in zip(target[0], mutant[0]):
            trial_nodes.append(m if random.random() < self.CR else t)
        return [trial_nodes, target[1]]

    def avaliar(self, individuo):
        pen = 0
        nodes, elements = individuo
        estrutura = Estrutura(elements, nodes,50,10,10)
        K_global, M_global = estrutura.matrizes_global()
        F_flexao1 = np.array([1000, 2000, 3000, 4000, 5000])
        F_flexao2 = np.array([1000, 1000, 1000, 1000, 1000])
        F_axial = np.array([1000, 2000, 3000, 4000, 5000])
        F_torcao = np.array([1000, 2000, 3000, 4000, 5000])
        _,_,_,_,_,KT,KF,_,_ = estrutura.shape_fun(F_flexao1, F_flexao2, F_axial, F_torcao)
        _, _, freq = estrutura.modal_analysis()
        fixed_dofs = [0, 1, 2, 3, 4, 5]
        F_global = np.zeros(estrutura.num_dofs)
        F_global[6*4 + 1] = 3000
        F_global[6*5 + 1] = 3000
        deslocamentos = estrutura.static_analysis(K_global, F_global, fixed_dofs)
        stresses = estrutura.compute_stress(estrutura.compute_strain(deslocamentos), 210e9, 0.27)
        von_mises = estrutura.compute_von_mises(stresses)
        massa = estrutura.mass()

        pen+=penalidade_chassi(KT,KF,massa,0)

        # simetria
        for i in range(0, len(nodes) - 1, 2):  # Stop one element earlier
            n1, n2 = nodes[i], nodes[i + 1]
            if not (
                np.isclose(n1[0], -n2[0], atol=0.05)
                and np.isclose(n1[1:], n2[1:], atol=0.05).all()
            ):
                pen += 1e4
        # comprimento longitudinal
        y_coords = [n[1] for n in nodes]
        if max(y_coords) - min(y_coords) < 1.6 or max(y_coords) - min(y_coords) > 2.0:
            pen += 1e6
        # conectividade e comprimento de elementos
        conex = [0] * len(nodes)
        for i, j in elements:
            ni, nj = nodes[i], nodes[j]
            d = np.linalg.norm(np.array(ni) - np.array(nj))
            if d < self.MIN_DIST or d > self.MAX_DIST:
                pen += 1e4
            conex[i] += 1
            conex[j] += 1
        pen += sum(1e4 for c in conex if c < 3)

        # Penaliza se algum elemento obrigatório for removido
        for e in self.subestrutura_obrigatoria['elements']:
            if e not in elements and (e[1], e[0]) not in elements:
                pen += 1e5  # Penalidade forte

        # Penaliza se os nós obrigatórios forem muito diferentes dos originais
        for i, ref_node in enumerate(self.subestrutura_obrigatoria['nodes']):
            dist = np.linalg.norm(np.array(nodes[i]) - np.array(ref_node))
            if dist > 0.05:
                pen += 1e3

        if len(nodes) == 0 or len(elements) == 0:
            return 1e9, None
        return (pen, massa,KT,KF)
    
    def evoluir(self):
        """
        Evolução diferencial com elitismo: preserva o melhor indivíduo.
        """
        for gen in range(self.NGEN):
            print(f"Geração {gen+1}/{self.NGEN}", end="\r")
            # guarda o melhor da população
            best_parent = min(self.population, key=lambda ind: self.avaliar(ind)[0])

            nova_pop = []
            for i, target in enumerate(self.population):
                idxs = list(range(self.POP_SIZE))
                idxs.remove(i)
                a, b, c = random.sample([self.population[k] for k in idxs], 3)
                mutant = self.mutacao(target, a, b, best_parent)
                trial = self.recombinacao(target, mutant)
                f_t = self.avaliar(target)[0]
                f_r = self.avaliar(trial)[0]
                nova_pop.append(trial if f_r < f_t else target)

            # elitismo: substitui o pior por best_parent
            worst_idx = max(range(len(nova_pop)), key=lambda i: self.avaliar(nova_pop[i])[0])
            nova_pop[worst_idx] = best_parent

            self.population = nova_pop

        # retorna o melhor final
        return min(self.population, key=lambda ind: self.avaliar(ind)[0])

    def plotar(self, individuo):
        nodes, elements = individuo
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = zip(*nodes)
        ax.scatter(ys, xs, zs, s=50)
        for i, j in elements:
            ni, nj = nodes[i], nodes[j]
            ax.plot([ni[1], nj[1]], [ni[0], nj[0]], [ni[2], nj[2]], 'b-')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        plt.title("Chassi Evoluído (DE com Elitismo)")
        plt.show()

def penalidade_chassi(KT, KF, massa, tensoes):
    KT_min = 1e8
    KF_min = 1e7
    massa_ideal = 23
    K_mola = 5e5
    lambdaP = 20  # Penalidade maior

    penalidade_total = 0

    if KT < KT_min:
        penalidade_total += lambdaP * ((KT_min - KT) / KT_min) ** 2  # Penalidade quadrática
    if KF < KF_min:
        penalidade_total += lambdaP * ((KF_min - KF) / KF_min) ** 2   # Penalidade quadrática
    if massa > massa_ideal:
        penalidade_total += lambdaP * ((massa - massa_ideal) / massa_ideal) ** 2
    if K_mola > 25*KF or K_mola > 25*KT:
        penalidade_total += 1e5  # Penalidade fixa severa

    return (penalidade_total)

if __name__ == "__main__":
    chassi = ChassiDE()
    best = chassi.evoluir()
    fitness, m, KT, KF= chassi.avaliar(best)
    nodes, elements = best    
# Exibindo as listas
    print("nodes = np.array([")
    for node in nodes:
      x, y, z = [float(coord) for coord in node]  # conversão explícita
      print(f"    [{x:.3f}, {y:.3f}, {z:.3f}],")
      print("\n")

    print("elements = [")
    for elem in elements:
        print(f"    {elem},")
    print("]")
    print(f"\nMelhor: massa={m:.1f} kg, fitness={fitness:.1f}, Rigidez Flexional={KF:.1f} Nm, Rigidez Torcional={KT:.1f} Nm/rad")
    chassi.plotar(best)
