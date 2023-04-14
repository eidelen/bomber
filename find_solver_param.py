from solver import apply_ppo
import numpy as np

def find_params():
    gamma = 0.7
    for i in range(0, 9):
        apply_ppo(gamma)
        gamma += 0.05


if __name__ == '__main__':
    find_params()