# Importing the packages we need
from types import SimpleNamespace
from scipy import optimize
import numpy as np
import sympy as sm
from ipywidgets import interact
import pandas as pd 
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class Solow:
    def __init__(self):

        # Setting up parameters
        # Defining namespaces
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # We start by setting up our algebraic parameters
        par.K = sm.symbols('k')
        par.A = sm.symbols('a')
        par.L = sm.symbols('L')
        par.Y = sm.symbols('Y')
        par.S = sm.symbols('S')
        
        par.alpha = sm.symbols('alpha')
        par.delta = sm.symbols('delta')
        par.s = sm.symbols('s')
        par.g = sm.symbols('g')
        par.n = sm.symbols('n')

        # We define our capital per effective worker
        par.k = sm.symbols('k')

        # Defining our sim variables
        sim.alpha = 0.3
        sim.delta = 0.05
        sim.s = 0.2  
        sim.g = 0.2
        sim.n = 0.02
        self.num_periods = 100
        self.k_0 = 1e-7

    def ss_equation(self):
        par = self.par
        # We define our transition equation
        f = par.k**par.alpha
        tranisition = sm.Eq(par.k, 1/((1+par.n)*(1+par.g))*((1-par.delta)*par.k + par.s*f))

        # We solve for the steady state
        steady_state = sm.solve(tranisition, par.k)[0]

        return steady_state
    
    def ss_value(self): 

        par = self.par
        sim = self.sim

        # We turn our symbolic steady state into a function
        ss_function = sm.lambdify((par.s,par.g,par.n,par.delta,par.alpha), self.ss_equation())

        return ss_function(sim.s,sim.g,sim.n,sim.delta,sim.alpha)
    
    def transition_diagram(self):

        capital_values = [self.k_0]  # List to store the capital stock values

        # Calculate the capital stock values for each period
        for _ in range(self.num_periods):
            k_t = capital_values[-1]  # Current capital stock
            k_t1 = 1 / ((1 + self.sim.n) * (1 + self.sim.g)) * (
                    self.sim.s * k_t ** self.sim.alpha + (1 - self.sim.delta) * k_t)
            capital_values.append(k_t1)

        # Transition diagram plotting
        plt.figure(figsize=(10, 6)) 
        plt.plot(capital_values[:-1], capital_values[1:], 'bo-', markersize=0, linewidth=1, label='k_t+1')
        plt.plot(capital_values[:-1], capital_values[:-1], 'ro-', markersize=0, linewidth=1, label='k_t = k_t+1')
        plt.xlabel('Capital level at t')
        plt.ylabel('Capital level at t+1')
        plt.title('Transition diagram of capital')
        plt.legend()
        plt.grid(True)
        plt.show()

    def interactive_plot(self):
        # FloatSliders for simulation variables
        a_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.3, description='Alpha')
        d_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.05, description='Delta')
        n_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.02, description='n')
        g_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.2, description='g')
        s_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.2, description='s')


        # Interactive update of simulation variables
        def update_sliders(alpha, delta, n, g, s):
            self.sim.alpha = alpha
            self.sim.delta = delta
            self.sim.n = n
            self.sim.g = g
            self.sim.s = s
            self.transition_diagram()

        interactive_plot = widgets.interactive(update_sliders,
                                               alpha=a_slider, delta=d_slider, n=n_slider, g=g_slider, s=s_slider)
        display(interactive_plot)    