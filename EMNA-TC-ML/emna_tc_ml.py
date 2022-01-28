import numpy as np
import sklearn
import pickle


def get_vector(data_matrix, max_f):
    # place two evaluation values
    x_vector = [data_matrix[0][0], data_matrix[1][0]]

    # Threshold values
    for i in range(0, len(data_matrix[3])):
        x_vector.append(data_matrix[3][i])

    # values between threshold and best_f
    for i in range(4, len(data_matrix)):
        for m in range(0, len(data_matrix[4])):
            x_vector.append(data_matrix[i][m] / max_f)

    x_vector = np.asarray(x_vector)
    x_vector = x_vector.reshape(1, -1)
    return x_vector


def emna_tc_ml(fun, dim, max_eval, pop_size, bound, gamma, model_path):
    # EMNA parameters
    sel_coef = 0.2
    gammas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    loaded_model = pickle.load(open(model_path, 'rb'))
    # Data
    # 0 - evaluations consumed
    # 1 - evaluations available
    # 2 - gamma
    # 3 - threshold
    # 4 - std dev fitness
    # 5 - average fitness
    # 6:16 - percentiles (0,10,20,30,...,100)
    data = np.zeros((17, 100))  # 200 every other iteration
    iter_i = 0


    # Initial population
    population = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    f_pop = fun(population)
    count_eval = pop_size

    best_idx = np.argmin(f_pop)
    best_f = f_pop[best_idx]
    best = population[best_idx, :]

    init_threshold = None
    remaining_eval = max_eval
    count_thre = count_eval
    max_f = np.max(f_pop)

    threshold_data = np.zeros(int((max_eval-count_eval)/pop_size))
    iterations = 0

    while count_eval < max_eval:
        arg_sorted = np.argsort(f_pop)
        ref_sols = population[arg_sorted[:int(pop_size*sel_coef)]]

        if init_threshold is None:
            init_threshold = np.linalg.norm(np.cov(ref_sols, rowvar=False))
        threshold = np.maximum(init_threshold * (np.power((remaining_eval - count_thre) / remaining_eval, gamma)), 1e-05)

        sigma = np.cov(ref_sols, rowvar=False)
        sigma = threshold * (sigma/np.linalg.norm(sigma))  # Applying TC
        miu = np.mean(ref_sols, axis=0)

        population = np.random.multivariate_normal(miu, sigma, pop_size)
        f_pop = fun(population)
        count_eval += pop_size
        count_thre += pop_size

        best_idx = np.argmin(f_pop)
        if best_f > f_pop[best_idx]:
            best_f = f_pop[best_idx]
            best = population[best_idx, :]

        # Collecting the data
        data[0, iter_i] = count_eval
        data[1, iter_i] = max_eval - count_eval
        data[2, iter_i] = gamma
        data[3, iter_i] = threshold
        data[4, iter_i] = np.std(f_pop)
        data[5, iter_i] = np.average(f_pop)
        data[6:17, iter_i] = np.percentile(f_pop, range(0, 110, 10))
        iter_i = (iter_i + 1) % 100

        if 0.1*max_eval < count_eval < 0.7*max_eval and iter_i % 10 == 0:
            data = np.roll(data, 100 - iter_i, axis=1)
            iter_i = 0
            x = get_vector(data, max_f)
            y = loaded_model.predict(x)
            gamma = gammas[y[0]]
            init_threshold = threshold
            count_thre = 0
            remaining_eval = max_eval - count_eval

        elif 0.7*max_eval < count_eval:
            gamma = 4

        threshold_data[iterations] = threshold
        iterations = iterations + 1

    return threshold_data, best_f
