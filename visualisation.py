import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class InteractivePlot():
    def __init__(self, model, xs, ys, config=dict()):
        self.model = model
        self.xs = xs
        self.ys = ys
        self.config = config
        self.ind = 1

        # set defaults in case user has not set them
        self.config["use_gpu"] = config.get("use_gpu", False)
        self.config["plot_innovation"] = config.get("plot_innovation", False)
        self.config["plot_particles"] = config.get("plot_particles", False)
        self.config["const_min_lim"] = config.get("const_min_lim", None)
        self.config["const_max_lim"] = config.get("const_max_lim", None)

        # uninitialized plot data (call init_plot to initialize)
        self.fig = None
        self.ax = None
        self.ys_pred_line = None
        self.ys_true_line = None
        self.xs_true_line = None
        self.particle_plot = None


    def init_plot(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ys_pred, ys_true, xs_true, particle_x, particle_y = self.create_plot_data(self.ind)

        self.ys_pred_line, = plt.plot(ys_pred, color="orange", label="pred")
        self.ys_true_line, = plt.plot(ys_true, color="blue", label="true")
        if self.config["plot_innovation"]:
            self.xs_true_line, = plt.plot(xs_true, color="pink",  label="innovation", linewidth=0.3)
        if self.config["plot_particles"]:
            self.particle_plot = plt.scatter(particle_x, particle_y, color="lightskyblue", label="particles", s=0.04)
        plt.legend(loc="upper left")

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)

        plt.show()


    def create_plot_data(self, series_num):

        single_series = self.xs[series_num:series_num+1]
        if torch.cuda.is_available() and self.config["use_gpu"]:
                    single_series = single_series.to('cuda')
        ys_pred, particle_pred = self.model.forward(single_series)

        # convert to numpy arrays
        ys_pred = ys_pred.cpu().detach().numpy()
        ys_true = self.ys.cpu().detach().numpy()
        particle_pred = particle_pred.cpu().detach().numpy()

        # flatten into a 1D array
        ys_pred = ys_pred.reshape((len(ys_pred), ))
        ys_true = ys_true[series_num].reshape((len(self.ys[series_num], )))
        xs_true = self.xs[series_num, :,-1].reshape((len(self.xs[series_num]), ))
        particle_pred_y = particle_pred.flatten()

        print("mse error: ", end="")
        print(sum([(ys_true[i] - ys_pred[i])**2 for i in range(len(ys_true))]) / len(ys_pred))

        # create particle pred y
        num_particles = particle_pred.shape[1]
        sequence_length = particle_pred.shape[0]
        particle_pred_x = np.zeros(num_particles * sequence_length)
        for xi in range(0, sequence_length):
            for pi in range(0, num_particles):
                particle_pred_x[xi * num_particles + pi] = xi

        return ys_pred, ys_true, xs_true, particle_pred_x, particle_pred_y
    

    # required function for button callback
    def next(self, event):
        self.ind += 1 
        self.update()


    # required function for button callback
    def prev(self, event):
        self.ind -= 1 
        self.update()


    # required function for button callback
    def update(self):
        i  = self.ind %(len(self.xs))
        ys_pred,ys_true,xs_true,particle_x, particle_y = self.create_plot_data(i) #unpack tuple data
        y_data = list(range(len(ys_pred)))
        self.ys_pred_line.set_ydata(ys_pred)
        self.ys_pred_line.set_xdata(y_data)
        self.ys_true_line.set_ydata(ys_true)
        self.ys_true_line.set_xdata(y_data)
        if self.config["plot_innovation"]:
            self.xs_true_line.set_ydata(xs_true)
            self.xs_true_line.set_xdata(y_data)
        if self.config["plot_particles"]:
            self.particle_plot.set_offsets(np.column_stack((particle_x, particle_y)))
        
        y_min = min(min(ys_pred), min(ys_true))
        y_max = max(max(ys_pred), max(ys_true))
        if self.config["plot_innovation"]:
            y_min = min(y_min, min(xs_true))
            y_max = max(y_max, max(xs_true))
        if self.config["plot_particles"]:
            y_min = min(y_min, min(particle_y))
            y_max = max(y_max, max(particle_y))
        if self.config["const_min_lim"]:
            y_min = self.config["const_min_lim"]
        if self.config["const_max_lim"]:
            y_max = self.config["const_max_lim"]
        self.ax.set_ylim(
            y_min, 
            y_max
        )
        plt.draw()

