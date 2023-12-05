from evotorch.algorithms import SteadyStateGA
import numpy as np

class SteadyStateGAWithEarlyStopping(SteadyStateGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_fitness_so_far = float('inf')
        self.generations_without_improvement = 0

    def run(self, n_generations):
        for _ in range(n_generations):
            current_best_fitness = super().run(1)  # Run for one generation
            status =super().status
            vals = []
            for i in range(5):
                if "obj"+str(i)+"_pop_best_eval" in status:
                    vals.append(status["obj"+str(i)+"_pop_best_eval"])
            if "pop_best_eval" in status:
                vals.append(status["mean_eval"])
            current_best_fitness=np.mean(vals)
            if current_best_fitness < self.best_fitness_so_far:
                self.best_fitness_so_far = current_best_fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

            if self.generations_without_improvement > 25:  # Stop if no improvement for 10 generations

                return self.best_fitness_so_far, _

        return self.best_fitness_so_far, 100