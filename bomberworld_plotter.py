# bomberworld_plotter.py
#
# Produces nice images of the Bomberworld environment
#
# Author: Giacomo Del Rio and extended by Adrian Schneider
# Creation date: 09 April 2023

from pathlib import Path
from typing import Tuple, List, Union, Optional
from PIL import Image
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

class BomberworldPlotter:

    def __init__(self, size: int, animated_gif_folder_path: Optional[Union[str, Path]] = None):
        """
        size: grid size of the bomberworld
        animated_gif_folder_path: if set, episodes are saved into this folder and
                                  calling create_animated_gif_from_episodes creates
                                  an animated gif of all saved episodes.
        """
        self.size = size
        self.animated_gif_folder = animated_gif_folder_path
        self.current_episode_nbr = 0
        self.ordered_file_list = []
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

    def plot_episode(self, current_reward = None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=mpl.figure.figaspect(1))

        self.draw_grid(ax)

        self.draw_stones(ax, self.stones)
        self.draw_path(ax, self.agent_traj, color='red', line_width=1)
        self.draw_bombs(ax, self.bomb_traj)
        self.draw_agent(ax, self.agent_traj[0][0], self.agent_traj[0][1])

        if current_reward is not None:
            ax.text(0.0, -1.0, f"Reward: {current_reward}")

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)

        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.tight_layout()

        if self.animated_gif_folder is not None:
            out_file = f"{self.animated_gif_folder}/{self.current_episode_nbr}.png"
            fig.savefig(out_file, dpi=200, bbox_inches='tight')
            self.ordered_file_list.append(out_file)
            self.current_episode_nbr += 1
        else:
            plt.show()

        plt.close()

    @staticmethod
    def draw_path(ax: mpl.axes.Axes, path: List[Tuple[int, int]], color, line_width):
        rng = default_rng()
        trj = [(c + rng.uniform(0.1, 0.4), r + rng.uniform(0.1, 0.4)) for r, c in path]
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

    def create_animated_gif_from_episodes(self):
        # creates animated gif
        # from https://pythonprogramming.altervista.org/png-to-gif/

        if self.animated_gif_folder is not None:
            frames = []
            for i in self.ordered_file_list:
                new_frame = Image.open(i)
                frames.append(new_frame)

            gif_out_path = f"{self.animated_gif_folder}/episode.gif"
            frames[0].save(gif_out_path, format='GIF',
                           append_images=frames[1:],
                           save_all=True,
                           duration=300, loop=0)
            print("Animated gif created, nbr imgs:", len(frames))
        else:
            print("Error: animated_gif_folder_path needs to be set in ctor")

