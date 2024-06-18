import numpy as np
import random

# Define the fitness function
def fitness_function(solution):
    # Example fitness function: Sphere function (minimize sum of squares)
    return -sum(x**2 for x in solution)  # Negative because we want to maximize

# Initialize the population
def initialize_population(pop_size, n_dims, bounds):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(bounds[dim][0], bounds[dim][1]) for dim in range(n_dims)]
        population.append(individual)
    return population

# Selection function: Tournament Selection
def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

# Crossover function: Single Point Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function: Gaussian Mutation
def mutate(individual, mutation_rate, bounds, mutation_strength=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal(0, mutation_strength)
            individual[i] = np.clip(individual[i], bounds[i][0], bounds[i][1])
    return individual

# Main Genetic Algorithm
def genetic_algorithm(pop_size, n_dims, bounds, generations, mutation_rate, elitism_rate=0.1):
    population = initialize_population(pop_size, n_dims, bounds)
    best_fitness = float('-inf')
    best_individual = None
    
    for generation in range(generations):
        fitnesses = [fitness_function(individual) for individual in population]
        
        # Selection
        selected_population = selection(population, fitnesses)
        
        # Crossover
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(child1)
            next_generation.append(child2)
        
        # Mutation
        next_generation = [mutate(individual, mutation_rate, bounds) for individual in next_generation]
        
        # Elitism
        num_elites = int(elitism_rate * pop_size)
        elites = sorted(list(zip(population, fitnesses)), key=lambda x: x[1], reverse=True)[:num_elites]
        next_generation = next_generation[:-num_elites] + [elite[0] for elite in elites]
        
        # Update population
        population = next_generation
        
        # Optionally, print the best solution in each generation
        best_fitness_in_gen = max(fitnesses)
        if best_fitness_in_gen > best_fitness:
            best_fitness = best_fitness_in_gen
            best_individual = population[fitnesses.index(best_fitness_in_gen)]
        
        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
    
    return best_individual, best_fitness

# Example usage
pop_size = 50
n_dims = 5
bounds = [(-10, 10) for _ in range(n_dims)]
generations = 100
mutation_rate = 0.1

best_solution, best_fitness = genetic_algorithm(pop_size, n_dims, bounds, generations, mutation_rate)
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)
