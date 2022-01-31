import functions
from emna_tc_data import emna_tc_data
import sys
from linked_node import LinkedNode
import numpy as np
import pickle

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

dim = 30
max_evals = 10_000 * dim
cec_benchmark = functions.CEC_functions(dim)


def obj_function(X):
    return cec_benchmark.Y_matrix(X, fun_num)


runs = 100  # _end = int(sys.argv[2])
'''#print(format('sys.argv[1]: '), sys.argv[1])
#test = 'sys.argv[1]: {}' # I added to test something
#print(test.format(sys.argv[0]))'''
exp = int(sys.argv[1])
for fun_num in range(1, 29):
    runs_list = []
    for run in range(runs):
        root = LinkedNode(None)
        dec_point = int(np.random.uniform(0.25 * max_evals, 0.95 * max_evals))
        emna_tc_data(obj_function, dim, max_evals, 500, 100, root, dec_point, None,
                     None, None, None, 0, 3)
        runs_list.append(root)

    pickle.dump(runs_list, open(f"fun{fun_num}_exp{exp}.p", "wb"), protocol=4)
