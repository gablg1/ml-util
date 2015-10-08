import matplotlib.pyplot as plt
import numpy as np

# Wrapper around matplotlib.pyplot that simplifies the easiest cases
class Plotter():
    def __init__(self, x_min=-1, x_max=1, nsamples=100):
        self.x_plot = np.linspace(x_min, x_max, nsamples)

    def plotData(self, x, y, *args, **kwargs):
        plt.scatter(x, y, *args, **kwargs)

    def plotFunction(self, f):
        vf = np.vectorize(f)
        y_plot = vf(self.x_plot)
        assert(self.x_plot.shape == y_plot.shape)
        plt.plot(self.x_plot, y_plot, self.args, self.kwargs)

    def show(self):
        plt.show()
