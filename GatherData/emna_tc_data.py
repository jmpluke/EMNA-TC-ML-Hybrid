import numpy as np
from linked_node import LinkedNode


def emna_tc_data(fun, dim, max_eval, pop_size, bound, parent_node, dec_point,
                 population, f_pop, best_f, init_threshold, start_eval, gamma):
    # EMNA parameters
    sel_coef = 0.2
    # gamma = 3

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
    gamma_param = gamma

    if start_eval == 0:
        # Initial population
        population = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
        f_pop = fun(population)
        count_eval = pop_size

        best_idx = np.argmin(f_pop)
        best_f = f_pop[best_idx]
        best = population[best_idx, :]
    else:
        count_eval = start_eval

    max_f = np.max(f_pop)
    collect = True

    while count_eval < max_eval:
        arg_sorted = np.argsort(f_pop)
        ref_sols = population[arg_sorted[:int(pop_size * sel_coef)]]

        if init_threshold is None:
            init_threshold = np.linalg.norm(np.cov(ref_sols, rowvar=False))
        threshold = np.maximum(init_threshold * (np.power((max_eval - count_eval) / max_eval, gamma)), 1e-05)

        sigma = np.cov(ref_sols, rowvar=False)
        sigma = threshold * (sigma / np.linalg.norm(sigma))  # Applying TC
        miu = np.mean(ref_sols, axis=0)

        population = np.random.multivariate_normal(miu, sigma, pop_size)
        f_pop = fun(population)
        count_eval += pop_size

        # Collecting the data
        if dec_point < max_eval:  # avoid data collection once it is done
            data[0, iter_i] = count_eval
            data[1, iter_i] = max_eval - count_eval
            data[2, iter_i] = gamma
            data[3, iter_i] = threshold
            data[4, iter_i] = np.std(f_pop)
            data[5, iter_i] = np.average(f_pop)
            data[6:17, iter_i] = np.percentile(f_pop, range(0, 110, 10))

            iter_i = (iter_i + 1) % 100

        if dec_point > max_eval and count_eval > start_eval + 0.1 * max_eval:
            gamma = 3

        # branching out
        if dec_point < count_eval:
            # max_f and init_t
            parent_node.max_f = max_f
            parent_node.init_t = init_threshold
            parent_node.data = np.roll(data, 100 - iter_i, axis=1)

            gammas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
            for g in gammas:
                emna_tc_data(fun, dim, max_eval, pop_size, bound, parent_node, np.inf,
                             population, f_pop, best_f, init_threshold, count_eval, g)
            return

        best_idx = np.argmin(f_pop)
        if best_f > f_pop[best_idx]:
            best_f = f_pop[best_idx]
            best = population[best_idx, :]

    current_node = LinkedNode(parent_node)
    current_node.evals = count_eval
    current_node.best_f = best_f
    current_node.gamma = gamma_param

    parent_node.results.append(current_node)

    return
