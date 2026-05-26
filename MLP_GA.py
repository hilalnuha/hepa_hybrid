import matplotlib.pyplot as plt

# 1. Initialize a list, best_fitness_history
best_fitness_history = []

print("Starting GA optimization...")

# 2. Start a loop for num_generations
for generation in range(num_generations):
    # 3. Calculate the fitness for each individual in the current population
    fitness_scores = np.array([calculate_fitness_ga(individual, X_train, y_train_one_hot) for individual in population])

    # 4. Update the best_individual_position and best_individual_fitness
    current_best_index = np.argmax(fitness_scores)
    current_best_fitness = fitness_scores[current_best_index]
    current_best_individual = population[current_best_index]

    if current_best_fitness > best_individual_fitness:
        best_individual_fitness = current_best_fitness
        best_individual_position = np.copy(current_best_individual)

    # 5. Append the current best_individual_fitness to the best_fitness_history list.
    best_fitness_history.append(best_individual_fitness)

    # 6. Perform tournament_selection to select parents for the next generation.
    parents = tournament_selection(population, fitness_scores)

    # 7. Create the next_population
    next_population = []
    for i in range(0, population_size, 2): # Take parents in pairs
        parent1 = parents[i]
        parent2 = parents[i+1] if (i+1) < population_size else parents[0] # Handle odd population size

        # Apply crossover
        offspring1, offspring2 = single_point_crossover(parent1, parent2, crossover_rate)

        # Apply mutation
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)

        next_population.extend([offspring1, offspring2])

    # Ensure next_population size matches population_size (in case of odd numbers or last pair)
    population = np.array(next_population[:population_size])

    # 9. Print the progress
    if generation % 10 == 0:
        print(f"Generation {generation}/{num_generations}, Best Fitness: {best_individual_fitness:.4f}")

print("GA optimization complete.")

# 10. Plot the best_fitness_history against the generations
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history)
plt.title('GA Best Fitness (Negative Loss) over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Negative Loss)')
plt.grid(True)
plt.show()

print("GA optimization curve plotted.")
