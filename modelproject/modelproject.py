import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import ipywidgets as widgets
import sympy as sm
from types import SimpleNamespace

class Solow:
    def __init__(self):

        # Setting up parameters and namespaces
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # Defining symbolic parameters and vairables
        par.A, par.Y, par.K, par.S, par.L, par.s, par.alpha, par.n, par.delta, par.k, par.g = sm.symbols('A Y K S L s alpha n delta k g')

        # Giving the parameters values
        sim.n = 0.02
        sim.alpha = 0.3 
        sim.g = 0.01
        sim.s = 0.2 
        sim.delta = 0.05
        self.num_periods = 99
        self.k_0 = 1e-7

    def ss_equation(self):
        # Defining the transition equation
        par = self.par
        f = par.k**par.alpha
        transition = sm.Eq(par.k, 1/((1+par.g)*(1+par.n))*(par.s*f + (1-par.delta)*par.k))
        # We find the ss
        ss = sm.solve(transition, par.k)[0]
        return ss
    
    def transition_diagram(self):
        capital_values = [self.k_0]
        par = self.par
        for _ in range(self.num_periods):
            k_t = capital_values[-1]
            k_t1 = ((1 - par['delta']) * k_t + par['s'] * (k_t ** par['alpha'])) / (1 + par['g'] + par['n'])
            capital_values.append(k_t1)

        plt.plot(capital_values[:-1], capital_values[1:], 'bo-', markersize=0, linewidth=1.5, label='k_t+1')
        plt.plot(capital_values[:-1], capital_values[:-1], 'ro-', markersize=0, linewidth=1.5, label='k_t = k_t+1')
        plt.xlabel('Capital level at t')
        plt.ylabel('Capital level at t+1')
        plt.title('Transition diagram of capital')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def interactive_plot(self):
        s_slider = widgets.FloatSlider(min=0, max=1, step=0.05, value=0.2, description='s')
        def s_slider_update(s):
            self.par['s'] = s
            self.transition_diagram()
        interactive_plot = widgets.interactive(s_slider_update, s=s_slider)
        display(interactive_plot)

class Solow_H:
    def __init__(self):
        # Setting up parameters and simulation values using dictionaries
        self.par = {
            'n': 0.02, 'delta': 0.05, 'g': 0.01, 
            's_H': 0.15, 'phi': 0.3, 's_K': 0.2, 'alpha': 0.3
        }
        self.num_periods = 99
        self.k_0 = 1e-7
        self.h_0 = 1e-7  # Initial value for human capital

    def calculate_steady_states(self):
        par = self.par
        # Simplified steady state calculations derived from the original steady state equations
        A = (1 + par['g']) * (1 + par['n'])
        B = par['delta'] + A - 1
        # Steady state human capital
        h_ss = (par['s_H'] / B) ** (1 / (1 - par['phi']))
        # Steady state physical capital
        k_ss = ((par['s_K'] * h_ss**par['phi']) / B) ** (1 / (1 - par['alpha']))

        return k_ss, h_ss

    def NClines(self):
        par = self.par
        k_ss, h_ss = self.calculate_steady_states()

        k_range = np.linspace(0, 20, 400)
        h_k = (par['s_H'] / (par['delta'] + par['g'] + par['n'])) ** (1 / (1 - par['phi'])) * k_range ** (par['alpha'] / (1 - par['phi']))
        k_h = ((par['s_K'] * k_range**par['alpha']) / (par['delta'] + par['g'] + par['n'])) ** (1 / (1 - par['phi']))

        plt.figure(figsize=(8, 6))
        plt.plot(k_range, h_k, label="Δh_t=0 (Human Capital Nullcline)")
        plt.plot(k_range, k_h, label="Δk_t=0 (Physical Capital Nullcline)")
        plt.plot(k_ss, h_ss, 'ro', label='Steady State')
        plt.xlabel('Physical Capital per Effective Worker (k_t)')
        plt.ylabel('Human Capital per Effective Worker (h_t)')
        plt.title('Phase Diagram')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.legend()
        plt.grid(True)
        plt.show()

    def interactive_NClines(self):
        par = self.par

        capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=par['s_K'], description='s_K')
        human_capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=par['s_H'], description='s_H')

        def update_phase_diagram(s_K, s_H):
            par['s_K'] = s_K
            par['s_H'] = s_H
            self.NClines()

        widgets.interact(update_phase_diagram, s_K=capital_savings_rate_slider, s_H=human_capital_savings_rate_slider)

