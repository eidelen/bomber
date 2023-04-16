# bomberworld_plotter.py
#
# Produces nice images of the Bomberworld environment
#
# Author: Giacomo Del Rio and modified Adrian Schneider
# Creation date: 09 April 2023

import itertools
from pathlib import Path
from typing import Tuple, List, Union, Optional

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


class GridworldPlotter:

    def __init__(self, size: int, walls: np.ndarray = None):
        self.size = size
        self.walls = walls

        self.agent_traj: List[Tuple[int, int]] = []
        self.bomb_traj: List[Tuple[int, int]] = []
        self.stones: np.ndarray = np.zeros((self.size, self.size), dtype=np.float32)
        self.agent_shape = [[.2, .6], [.2, .3], [.3, .1], [.7, .1], [.8, .3], [.8, .6]]

    def add_frame(self, agent_position: Tuple[int, int], bombed: bool, stones: np.ndarray ) -> None:
        if bombed:
            self.bomb_traj.append(agent_position)
        else:
            self.agent_traj.append(agent_position)

        self.stones = stones

    def plot_episode(self, out_file: Optional[Union[str, Path]] = None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=mpl.figure.figaspect(1))

        self.draw_grid(ax)

        self.draw_stones(ax, self.stones)
        self.draw_path(ax, self.agent_traj, color='red', line_width=1)
        self.draw_bombs(ax, self.bomb_traj)
        self.draw_agent(ax, self.agent_traj[0][0], self.agent_traj[0][1])

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.tight_layout()

        if out_file is not None:
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_rollouts(self, rollouts: List[List[Tuple[int, int]]], agent_start_pos: Tuple[int, int],
                      episode_trajectory: Optional[List[Tuple[int, int]]],
                      out_file: Union[str, Path]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=mpl.figure.figaspect(1))

        self.draw_grid(ax)
        for r in rollouts:
            self.draw_path(ax, r, color='darkred', line_width=0.5)
        if episode_trajectory is not None:
            self.draw_path(ax, episode_trajectory, color='c', line_width=2)
        self.draw_agent(ax, agent_start_pos[0], agent_start_pos[1])

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.tight_layout()

        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_path(ax: mpl.axes.Axes, path: List[Tuple[int, int]], color, line_width):
        rng = default_rng()
        trj = [(c + rng.uniform(0.3, 0.7), r + rng.uniform(0.3, 0.7)) for r, c in path]
        ax.add_patch(patches.Polygon(trj, closed=False, ec=color, lw=line_width, fill=False))

    @staticmethod
    def draw_bombs(ax: mpl.axes.Axes, bombs: List[Tuple[int, int]]):
        index = 0
        for m, n in bombs:
            ax.add_patch(patches.Ellipse((n+0.5, m+0.5), width=0.8, height=0.8, ec="black", fill=False))
            ax.text(n+0.3, m+0.6, str(index))
            index += 1


    @staticmethod
    def draw_stones(ax: mpl.axes.Axes, stones: np.ndarray):
        ms, ns = stones.shape
        for m in range(0, ms):
            for n in range(0, ns):
                if stones[(m,n)] < 0.1:
                    ax.add_patch(patches.Rectangle((n+0.125, m+0.125), width=0.75, height=0.75, ec='black', fc='grey', fill=True))

    def draw_grid(self, ax: mpl.axes.Axes):
        for i in range(self.size + 1):
            ax.axhline(y=i, c='k', lw=2)
            ax.axvline(x=i, c='k', lw=2)

    def draw_agent(self, ax: mpl.axes.Axes, row: int, col: int):
        agent_shape = [(r + col, c + row) for r, c in self.agent_shape]
        ax.add_patch(patches.Polygon(agent_shape, closed=True, ec='k', fc='c'))
        ax.add_patch(patches.Rectangle((.3 + col, .6 + row), .1, .3, ec='k', fc='c'))
        ax.add_patch(patches.Rectangle((.6 + col, .6 + row), .1, .3, ec='k', fc='c'))
        ax.add_patch(patches.Rectangle((.32 + col, .25 + row), .1, .15, ec='k', fc='w'))
        ax.add_patch(patches.Rectangle((.57 + col, .25 + row), .1, .15, ec='k', fc='w'))

    @staticmethod
    def draw_goal(ax: mpl.axes.Axes, row: int, col: int):
        ax.add_patch(patches.Rectangle((.3 + col, .5 + row), .07, .4, ec='k', fc='y'))
        ax.add_patch(
            patches.Polygon([(.3 + col, .15 + row), (.3 + col, .6 + row), (.7 + col, .35 + row)],
                            closed=True, ec='k', fc='y'))
