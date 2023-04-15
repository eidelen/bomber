from solver import apply_ppo

def find_params():
    gammas = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
    nn_models = [[256, 256]]
    activations = ["relu"]  # tanh, linear

    for nn in nn_models:
        for act in activations:
            for g in gammas:
                apply_ppo(g, nn, act)

if __name__ == '__main__':
    find_params()