from emna_tc_ml import emna_tc_ml
import functions
import pickle
import numpy as np


class OptResults(object):
    def __init__(self):
        self.fitness = []
        self.threshold = []


def obj_function(X):
    return cec_benchmark.Y_matrix(X, fun_num)


fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

dim = 30
max_evals = 10_000 * dim
cec_benchmark = functions.CEC_functions(dim)
popsize = 1000
model = "models/model_RNN_v2_5p.p"

#results = OptResults()
runs = 30
thresholds = []
fits = []

for run in range(runs):
    for fun_num in range(1, 29):
        threshold, best_f = emna_tc_ml(obj_function, dim, max_evals, popsize, 100, 3, model)
        if run == 0:
            fits.append([best_f - fDeltas[fun_num-1]])
            thresholds.append([threshold])
        else:
            fits[fun_num - 1].append(best_f - fDeltas[fun_num-1])
            thresholds[fun_num - 1].append(threshold)
        print(f"Run: {run}, EMNA-TC-ML Function {fun_num}, pop {popsize}, result: {(best_f - fDeltas[fun_num - 1]):.2E}")

pickle.dump(fits, open("pickles/results_RNN_v2_5p_fits.p", "wb"), protocol=4)
pickle.dump(thresholds, open("pickles/results_RNN_v2_5p_thresholds.p", "wb"), protocol=4)
print('fitnesses:')
print(fits)