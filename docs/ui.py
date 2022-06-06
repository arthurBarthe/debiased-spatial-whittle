import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


import numpy as np
from numpy.fft import fftshift
from debiasedwhittle import exp_cov2, sim_circ_embedding, fit, matern, periodogram, compute_ep

cov = matern

class App(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.wm_title("Spatial Debiased Whittle")
        self.root = tkinter.Frame(self)
        self.tk.call("source", "../azure.tcl")
        self.tk.call("set_theme", "../theme/dark")
        self.z = None
        self.cov = None
        self.g = None
        self.plot_names = ['data', 'periodogram', 'e_periodogram']
        self.plot_vars = dict()
        self.create_top_frame()
        self.create_middle_frame()
        self.root.pack()
        self.mainloop()

    @property
    def shape(self):
        m, n = self.m_entry.get(), self.n_entry.get()
        try:
            m, n = int(m), int(n)
        except ValueError:
            m, n = 128, 128
        return m, n

    @property
    def rho(self):
        try:
            return float(self.rho_entry.get())
        except ValueError:
            return 10.

    @property
    def nu(self):
        try:
            return float(self.nu_entry.get())
        except ValueError:
            return 2.5

    def create_top_frame(self):
        top_frame = tkinter.Frame(self.root)
        # Simulation button
        b = ttk.Button(text='New simulation')
        b.pack()
        b.config(command=self.update_data)

        # simulation parameters
        x_label = ttk.Label(top_frame, text="m")
        y_label = ttk.Label(top_frame, text="n")
        m_entry, n_entry = ttk.Entry(top_frame), tkinter.Entry(top_frame)
        x_label.pack(side=tkinter.LEFT)
        m_entry.pack(side=tkinter.LEFT)
        y_label.pack(side=tkinter.LEFT)
        n_entry.pack(side=tkinter.LEFT)

        rho_label = ttk.Label(top_frame, text='rho')
        rho_entry = ttk.Entry(top_frame)
        rho_label.pack(side=tkinter.LEFT)
        rho_entry.pack(side=tkinter.LEFT)

        nu_label = ttk.Label(top_frame, text='nu')
        nu_entry = ttk.Entry(top_frame)
        nu_label.pack(side=tkinter.LEFT)
        nu_entry.pack(side=tkinter.RIGHT)
        self.m_entry, self.n_entry = m_entry, n_entry
        self.rho_entry, self.nu_entry= rho_entry, nu_entry
        top_frame.pack()

    def create_middle_frame(self):
        middle_frame = tkinter.Frame(self.root)
        for p_name in self.plot_names:
            self.plot_vars[p_name] = self._make_matplotlib_canvas(middle_frame)
            self.plot_vars[p_name][0].get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        middle_frame.pack()

    def _make_matplotlib_canvas(self, parent_frame):
        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        return canvas, ax

    def update_data(self):
        self.cov_func = lambda lags: cov(lags, self.rho, self.nu, 1)
        # simulation
        self.g = np.ones(self.shape)
        self.z = sim_circ_embedding(self.cov_func, self.shape)
        self.update_plot()

    def update_plot(self):
        for name, (canvas, ax) in self.plot_vars.items():
            plot_func = getattr(self, 'plot_' + name)
            plot_func(ax)
            canvas.draw()

    def plot_data(self, ax):
        ax.imshow(self.z, cmap='coolwarm', origin='lower')

    def plot_periodogram(self, ax):
        e_per = compute_ep(self.cov_func, self.g)
        vmin, vmax = np.min(10*np.log10(e_per)), np.max(10*np.log10(e_per))
        per = periodogram(self.z)
        ax.imshow(fftshift(10 * np.log10(per)), vmin=vmin, vmax=vmax)

    def plot_e_periodogram(self, ax):
        e_per = compute_ep(self.cov_func, self.g)
        ax.imshow(fftshift(10 * np.log10(e_per)))



if __name__ == '__main__':
    app = App()
