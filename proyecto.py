import pygame
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
ITEMS = [15, 10, 5, 7, 9, 4, 6, 8, 3, 11, 2, 13, 14, 1, 12] *7 # Tamaño de los objetos
BIN_CAPACITY = 30  # Capacidad máxima de cada contenedor
ideal = np.ceil(np.sum(ITEMS) /BIN_CAPACITY)

SCREEN_WIDTH, SCREEN_HEIGHT = 1800, 800
BOX_WIDTH = 100
BOX_HEIGHT_UNIT = 20
MARGIN = 10
PASTEL_COLORS = [
    (244, 164, 96), (135, 206, 250), (152, 251, 152), 
    (255, 182, 193), (240, 230, 140), (221, 160, 221), 
    (176, 224, 230), (255, 218, 185), (144, 238, 144)
]
COLORS = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(ITEMS))]

# Inicialización de Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Bin Packing Algorithm Visualization")
font = pygame.font.SysFont("Arial", 18)
scroll_x, scroll_y = 0, 0
scroll_speed = 500

def plot_fitness_evolution(historic_best):
    """
    Grafica la evolución de la variable fitness almacenada en la lista historic_best.

    :param historic_best: Lista con los valores de fitness en cada generación.
    """
    if not historic_best:
        print("La lista historic_best está vacía. No hay datos para graficar.")
        return

    # Crear un rango para el eje x basado en el número de generaciones
    generations = range(len(historic_best))

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(generations, historic_best, marker='o', linestyle='-', color='b', label='Mejor fitness')

    # Etiquetas y título
    plt.title('Evolución del Fitness a lo Largo de las Generaciones', fontsize=14)
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

# Dibujar contenedores
def draw_bins(bins):
    global scroll_x, scroll_y
    screen.fill((255, 255, 255))  # Fondo blanco
    x_offset = MARGIN - scroll_x
    y_offset = SCREEN_HEIGHT - MARGIN - scroll_y

    for i, bin in enumerate(bins):
        x = x_offset + i * (BOX_WIDTH + MARGIN)
        y = y_offset
        for item in bin:
            color = COLORS[ITEMS.index(item)]  # Asignar color basado en índice
            rect = pygame.Rect(x, y - item * BOX_HEIGHT_UNIT, BOX_WIDTH, item * BOX_HEIGHT_UNIT)
            pygame.draw.rect(screen, color, rect, border_radius=5)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=5)  # Borde negro
            
            # Mostrar el tamaño del objeto
            label = font.render(str(item), True, (0, 0, 0))  # Texto negro
            label_rect = label.get_rect(center=(x + BOX_WIDTH // 2, y - (item * BOX_HEIGHT_UNIT // 2)))
            screen.blit(label, label_rect)
            
            y -= item * BOX_HEIGHT_UNIT
        
        # Mostrar el número del contenedor
        label = font.render(f"Bin {i+1}", True, (0, 0, 0))
        screen.blit(label, (x + BOX_WIDTH // 4, SCREEN_HEIGHT - 30 - scroll_y))

# Función de fitness
def fitness(solution):
    nb = len(solution)  # Número de contenedores
    penalty = 0
    for bin in solution:
        fk = sum(bin)
        penalty += (fk / BIN_CAPACITY) ** 2
    return nb - penalty

# Inicializar población
def initialize_population(items, bin_capacity, population_size):
    population = []
    for _ in range(population_size):
        shuffled_items = random.sample(items, len(items))
        bins = pack_items(shuffled_items, bin_capacity)
        population.append(bins)
    return population

# Empaquetar ítems en contenedores
def pack_items(items, bin_capacity):
    bins = []
    for item in items:
        placed = False
        for bin in bins:
            if sum(bin) + item <= bin_capacity:
                bin.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])
    return bins

# Selección por truncación
def truncation_selection(population, truncation_percentage):
    population.sort(key=fitness)
    cutoff = int(len(population) * truncation_percentage)
    return random.choice(population[:cutoff])

# Construir y actualizar RPFM
# Se agrega la actualización incremental usando ΔRPFM y un factor de relajación α
def build_rpfm(population, items, rpfm, alpha=0.2):
    size = len(items)
    delta_rpfm = [[0] * size for _ in range(size)]

    # Seleccionar los mejores individuos (20% de la población)
    best_individuals = sorted(population, key=fitness)[:int(0.2 * len(population))]

    # Calcular ΔRPFM
    for individual in best_individuals:
        for bin in individual:
            if len(bin) > 1:  # Solo considerar bins con más de un ítem
                for i in range(len(bin)):
                    for j in range(i + 1, len(bin)):
                        index_i = ITEMS.index(bin[i])
                        index_j = ITEMS.index(bin[j])
                        delta_rpfm[index_i][index_j] += 1
                        delta_rpfm[index_j][index_i] += 1

    # Normalizar ΔRPFM dividiéndolo por la cantidad de mejores individuos
    num_best_individuals = len(best_individuals)
    for i in range(size):
        for j in range(size):
            if num_best_individuals > 0:
                delta_rpfm[i][j] /= num_best_individuals

    # Actualizar RPFM usando ΔRPFM y el factor de relajación α
    for i in range(size):
        for j in range(size):
            rpfm[i][j] = (1 - alpha) * rpfm[i][j] + alpha * delta_rpfm[i][j]

    return rpfm

# Mutación usando RPFM
def mutate_shuffle(solution, bin_capacity):
    items_flat = [item for bin in solution for item in bin]
    random.shuffle(items_flat)
    new_solution = pack_items(items_flat, bin_capacity)
    return new_solution

# Se basa en la probabilidad de pares en el RPFM para agrupar ítems relacionados en el mismo bin
def mutate_with_rpfm(solution, rpfm, bin_capacity, mutation_rate=1):
    if random.random() < mutation_rate:
        items_flat = [item for bin in solution for item in bin]
        new_bins = []

        # Crear un bin inicial con el primer ítem al azar
        remaining_items = items_flat[:]
        current_bin = [remaining_items.pop(random.randint(0, len(remaining_items) - 1))]

        while remaining_items:
            # Ordenar los ítems restantes según la relación con el último ítem en el bin actual
            last_item = current_bin[-1]
            related_items = sorted(
                remaining_items,
                key=lambda x: rpfm[ITEMS.index(last_item)][ITEMS.index(x)],
                reverse=True
            )

            # Intentar añadir el ítem más relacionado al bin actual
            for item in related_items:
                if sum(current_bin) + item <= bin_capacity:
                    current_bin.append(item)
                    remaining_items.remove(item)
                    break
            else:
                # Si no se puede añadir más, guardar el bin actual y empezar uno nuevo
                new_bins.append(current_bin)
                current_bin = [remaining_items.pop(0)]

        # Añadir el último bin
        new_bins.append(current_bin)
        if random.random() < 0.6 :
            new_bins = mutate_shuffle(new_bins, bin_capacity)
        return new_bins

    return solution

# Reinserción con elitismo
def reinsertion_with_elitism(new_population, old_population, elite_size):
    combined = new_population + old_population
    combined.sort(key=fitness)
    elites = combined[:elite_size]
    return elites + random.sample(combined[elite_size:], len(old_population) - elite_size)


# LS1: Intercambiar pares de ítems de FB con 1-3 ítems de NFB
def ls1(bins, bin_capacity):
    # Separar FB y NFB
    fb_bins = [bin for bin in bins if sum(bin) == bin_capacity]  # Fully filled bins
    nfb_bins = [bin for bin in bins if sum(bin) < bin_capacity]  # Not fully filled bins

    # Crear copia de bins para trabajar
    vitem = [item for bin in nfb_bins for item in bin]  # Vector de ítems NFB

    # Intentar intercambios
    for fb_bin in fb_bins:
        fb_bin_copy = fb_bin.copy()  # Copiar para trabajar sin modificar directamente
        if len(fb_bin_copy) < 2:
            continue  # No es posible intercambiar si hay menos de 2 ítems

        for i in range(len(fb_bin_copy)):
            for j in range(i + 1, len(fb_bin_copy)):  # Evitar combinaciones repetidas
                for k in range(1, 4):  # Probar intercambios con 1, 2 o 3 ítems de NFB
                    if len(vitem) >= k:
                        nfb_subset = vitem[:k]
                        # Verificar si el intercambio mantiene el FB completamente lleno
                        new_fb_bin = fb_bin_copy.copy()
                        if fb_bin_copy[i] in new_fb_bin and fb_bin_copy[j] in new_fb_bin:
                            new_fb_bin.remove(fb_bin_copy[i])
                            new_fb_bin.remove(fb_bin_copy[j])
                            new_fb_bin.extend(nfb_subset)

                            if sum(new_fb_bin) == bin_capacity:
                                # Aplicar intercambio
                                for nfb_item in nfb_subset:
                                    vitem.remove(nfb_item)
                                vitem.append(fb_bin_copy[i])
                                vitem.append(fb_bin_copy[j])
                                fb_bin[:] = new_fb_bin
                                break

    # Reconstruir NFB bins desde VITEM
    nfb_bins = []
    current_bin = []
    for item in vitem:
        if sum(current_bin) + item <= bin_capacity:
            current_bin.append(item)
        else:
            nfb_bins.append(current_bin)
            current_bin = [item]
    if current_bin:
        nfb_bins.append(current_bin)

    # Combinar FB y NFB
    return fb_bins + nfb_bins



# LS2: Intercambiar pares de ítems de FB con 1 ítem de NFB
def ls2(bins, bin_capacity):
    # Implementación de LS2 aquí
    return bins

# LS3: Intercambiar pares de ítems de FB con 2 ítems de NFB
def ls3(bins, bin_capacity):
    # Implementación de LS3 aquí
    return bins

# LS4: Intercambiar ítems de FB con 2 ítems de NFB
def ls4(bins, bin_capacity):
    # Implementación de LS4 aquí
    return bins

# LS5: Mejorar bins de FB insertando ítems desde NFB
def ls5(bins, bin_capacity):
    # Implementación de LS5 aquí
    return bins

# LS6: Mejorar bins de NFB insertando ítems desde otros NFB
def ls6(bins, bin_capacity):
    # Implementación de LS6 aquí
    return bins

# LS7: Reorganizar pares de bins de NFB para aumentar su llenado
def ls7(bins, bin_capacity):
    # Implementación de LS7 aquí
    return bins

# LS8: Intercambiar pares de bins vecinos en NFB
def ls8(bins, bin_capacity):
    # Implementación de LS8 aquí
    return bins

# LS9: Mejorar un bin aleatorio de NFB mediante intercambios
def ls9(bins, bin_capacity):
    # Implementación de LS9 aquí
    return bins

# LS10: Mejorar fitness mediante intercambios específicos en NFB
def ls10(bins, bin_capacity):
    # Implementación de LS10 aquí
    return bins

# Aplicar grupo de búsquedas locales
def apply_local_searches(bins, bin_capacity, apply_to_best=False):
    '''
    if apply_to_best:
        local_search_group = [ls1, ls2, ls3, ls4, ls7, ls9]
    else:
        local_search_group = [ls1, ls2, ls4, ls6, ls7, ls8, ls9, ls10]

    for ls in local_search_group:
        bins = ls(bins, bin_capacity)
    '''
    return bins


# Modificar el algoritmo genético para integrar las búsquedas locales
def genetic_algorithm_with_visualization(items, bin_capacity, population_size, generations, truncation_percentage, elite_size):
    global scroll_x, scroll_y
    population = initialize_population(items, bin_capacity, population_size)
    rpfm = [[0] * len(items) for _ in range(len(items))]
    rpfm = build_rpfm(population, items, rpfm)
    best_solution = min(population, key=fitness)
    gens = 1
    historic_best = []
    historic_best.append(fitness(best_solution))

    running = True
    for generation in range(generations):
        if not running or len(best_solution) == ideal:
            break
        gens = gens + 1
        new_population = []
        for _ in range(population_size):
            parent = truncation_selection(population, truncation_percentage)
            offspring = mutate_with_rpfm(parent.copy(), rpfm, bin_capacity)
            offspring = apply_local_searches(offspring, bin_capacity)
            new_population.append(offspring)

        population = reinsertion_with_elitism(new_population, population, elite_size)
        rpfm = build_rpfm(population, items, rpfm)
        best_candidate = min(population, key=fitness)
        if fitness(best_candidate) < fitness(best_solution):
            best_solution = apply_local_searches(best_candidate, bin_capacity, apply_to_best=True)

        # Dibujar la generación actual
        draw_bins(best_solution)
        label = font.render(f"Generation: {generation + 1}", True, (0, 0, 0))
        screen.blit(label, (MARGIN - scroll_x, MARGIN - scroll_y))
        pygame.display.flip()

        # Controlar eventos de salida
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    scroll_y = max(0, scroll_y - scroll_speed)
                elif event.key == pygame.K_DOWN:
                    scroll_y += scroll_speed
                elif event.key == pygame.K_LEFT:
                    scroll_x = max(0, scroll_x - scroll_speed)
                elif event.key == pygame.K_RIGHT:
                    scroll_x += scroll_speed

        # Pausa para visualizar el progreso
        historic_best.append(fitness(best_solution))
        time.sleep(0.05)

    print(f"Terminado en la generación: {gens}")
    plot_fitness_evolution(historic_best)
    return best_solution

# Parámetros del algoritmo genético
population_size = 200
generations = 500
truncation_percentage = 0.6
elite_size = 8

# Ejecutar el algoritmo
best_solution = genetic_algorithm_with_visualization(
    items=ITEMS,
    bin_capacity=BIN_CAPACITY,
    population_size=population_size,
    generations=generations,
    truncation_percentage=truncation_percentage,
    elite_size=elite_size,
)

# Resultado final
print(f"Mejor solución encontrada utiliza {len(best_solution)} contenedores:")
print(best_solution)

# Finalizar Pygame
pygame.quit()

