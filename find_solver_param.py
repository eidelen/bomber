from solver import apply_ppo

def find_params():
    gammas = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
    for g in gammas:
        apply_ppo(g, [256, 128, 64])

if __name__ == '__main__':
    find_params()