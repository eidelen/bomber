from solver import apply_ppo

# notes on good performing model
# [512, 256, 128, 64], Gamma = 0.8, relu: -> able to finish game
# [256, 256, 128, 64], Gamma = 0.8, relu: -> good score, but never finished game

def find_params():
    gammas = [0.80]
    nn_models = [[512, 256, 128, 64]]
    activations = ["relu"]  # tanh, linear

    for nn in nn_models:
        for act in activations:
            for g in gammas:
                apply_ppo(g, nn, act)

if __name__ == '__main__':
    find_params()