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
model0 = "models/model_RNN_org.p" # fun 1/2
model1 = "models/model_QDA_v2_1p.p" # fun 3
model2 = "models/model_RNN_v2_5p.p" # fun 4/5/13/24/25/28
model3 = "models/model_RNN_v2_1p.p" # fun 6/14/15
model4 = "models/model_RNN_v2_9p.p" # fun 7/10/16/17/18/19/22/23/26
model5 = "models/model_LDA_SVD_v2_9p.p" # fun 8
model6 = "models/model_DT_v3_1p.p" # fun 9
model7 = "models/model_DT_v2_5p.p" # fun 11/12
model8 = "models/model_QDA_org.p" # fun 20/27
model9 = "models/model_LDA_SVD_v2_5p.p" # fun 21

model = "models/model_RNN_org.p" # fun 1/2

#results = OptResults()
runs = 30
thresholds = []
fits = []

for run in range(runs):
    for fun_num in range(1, 29):

        if fun_num == 1 or fun_num == 2:
            model = model0
        elif fun_num == 3:
            model = model1
        elif fun_num == 4 or fun_num == 5 or fun_num == 13 or fun_num == 24 or fun_num ==25 or fun_num == 28:
            model = model2
        elif fun_num == 6 or fun_num == 14 or fun_num == 15:
            model = model3
        elif fun_num == 7 or fun_num == 10 or fun_num == 16 or fun_num == 17 or fun_num == 18 or fun_num == 19 or fun_num == 22 or fun_num == 23 or fun_num == 26:
            model = model4
        elif fun_num == 8:
            model = model5
        elif fun_num == 9:
            model = model6
        elif fun_num == 11 or fun_num == 12:
            model = model7
        elif fun_num == 20 or fun_num == 27:
            model = model8
        elif fun_num == 21:
            model = model9

        threshold, best_f = emna_tc_ml(obj_function, dim, max_evals, popsize, 100, 3, model)
        if run == 0:
            fits.append([best_f - fDeltas[fun_num-1]])
            thresholds.append([threshold])
        else:
            fits[fun_num - 1].append(best_f - fDeltas[fun_num-1])
            thresholds[fun_num - 1].append(threshold)
        print(f"Run: {run}, EMNA-TC-ML Function {fun_num}, pop {popsize}, result: {(best_f - fDeltas[fun_num - 1]):.2E}")

pickle.dump(fits, open("pickles/combo_fits.p", "wb"), protocol=4)
pickle.dump(thresholds, open("pickles/combo_thresholds.p", "wb"), protocol=4)
print('fitnesses:')
print(fits)