import torch
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
#from umap.parametric_umap import ParametricUMAP
from sklearn.decomposition import PCA
from celluloid import Camera

class Visualizer():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.colors = ['black', 'red', 'green', 'mediumblue', 'lightblue', 'yellow', 'orange', 'magenta']
        self.train_losses, self.eval_losses, self.train_penalties = [], [], []
        self.cmap = "RdYlGn"
        self._iter = 0
        sns.set()

    def plot_losses(self):
        l = len(self.train_losses)
        x_train = torch.linspace(0, l, l)
        x_eval = torch.linspace(0, l, len(self.eval_losses))
        plt.plot(x_train, self.train_losses, label='train loss')
        plt.plot(x_eval, self.eval_losses, label='eval loss')
        evalloss = min(self.eval_losses)
        plt.title(f'Iter {self._iter} ; Best Eval Loss : {str(evalloss)[:6]}')
        plt.axhline(evalloss, linestyle='--')
        plt.yscale('log')
        plt.legend()
        plt.savefig(self.log_dir+'/visu/losses'+str(self._iter)+'.png')
        plt.close()