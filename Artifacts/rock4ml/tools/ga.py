import math
import random

import numpy as np

def encode(combination, all_features_list):
  encoded = [0] * len(all_features_list)
  for feature in combination:
      if feature in all_features_list:
          index = all_features_list.index(feature)
          encoded[index] = 1
  return encoded

def decode(encoded, all_features_list):
  return [feature for feature, is_selected in zip(all_features_list, encoded) if is_selected]
  
def crossover(parent1, parent2, random_state):
  np.random.seed(random_state)
  # point = math.floor(len(parent1) / 2)
  point = np.random.randint(1, len(parent1)-1)
  offspring1 = parent1[:point] + parent2[point:]
  offspring2 = parent2[:point] + parent1[point:]
  return offspring1, offspring2

def mutate(individual, mutation_rate, random_state):
  np.random.seed(random_state)
  random_array = np.random.rand(len(individual))

  for i in range(len(individual)):
    if random_array[i] < mutation_rate:
      individual[i] = 1 if individual[i] == 0 else 0
  return individual

def geneticAlgorithm(parent1, parent2, random_state, mutation_rate=0.1):
  offspring1, offspring2 = crossover(parent1, parent2, random_state)

  offspring1 = mutate(offspring1, mutation_rate, random_state)
  offspring2 = mutate(offspring2, mutation_rate, random_state)

  return offspring1, offspring2

def updateCandidateFeatures(candidate_features_scores, topK, mutation_rate, gama, avail_features, random_state):
  candidate_features_combs = [candidate_features_scores[i][0] for i in range(min(topK, len(candidate_features_scores)))]
  candidate_features_offsprings = set()
  # print(candidate_features_combs)
  cnt = 0
  # print(avail_features)
  while len(candidate_features_offsprings) < gama:
    # random choose 2 number simultaneously
    np.random.seed(random_state+cnt)
    indices = np.random.choice(len(candidate_features_combs), size=2, replace=False)
    
    parent1, parent2 = candidate_features_combs[indices[0]], candidate_features_combs[indices[1]]
    
    parent1 = encode(parent1, avail_features)
    parent2 = encode(parent2, avail_features)

    offspring1, offspring2 = geneticAlgorithm(parent1, parent2, random_state+cnt, mutation_rate)
    
    offspring1 = decode(offspring1, avail_features)
    offspring2 = decode(offspring2, avail_features)

    if tuple(offspring1):
      candidate_features_offsprings.add(tuple(offspring1))

    if tuple(offspring2):
      candidate_features_offsprings.add(tuple(offspring2))

    cnt += 1

    if cnt >= 200:
      break

  # change each element in candidate_features_offsprings to list
  candidate_features_offsprings = sorted(list(candidate_features_offsprings))
  candidate_features_offsprings = [list(offspring) for offspring in candidate_features_offsprings]
  
  return candidate_features_offsprings
