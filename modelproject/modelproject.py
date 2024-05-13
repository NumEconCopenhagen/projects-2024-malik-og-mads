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

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from types import SimpleNamespace

class Solow_H:
    def __init__(self):
        # Setting up parameter namespaces
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        # Defining parameters
        self.sim.n = 0.02
        self.sim.delta = 0.05
        self.sim.g = 0.01
        self.sim.s_H = 0.15
        self.sim.phi = 0.3
        self.sim.s_K = 0.2
        self.sim.alpha = 0.3

    def ss_value(self, x):
        k, h = x
        alpha = self.sim.alpha
        delta = self.sim.delta
        s_K = self.sim.s_K
        s_H = self.sim.s_H
        g = self.sim.g
        n = self.sim.n
        phi = self.sim.phi

        # Define equations
        f1 = h - 1/((1+g) * (1+n)) * ((1-delta)*h + (s_H) * h**phi * k**alpha)
        f2 = k - 1/((1+g) * (1+n)) * ((1 - delta)*k + (s_K) * h**phi * k**alpha)

        return [f1, f2]

    def solve_numerical(self):
        # Initial guess
        x0 = [20, 20]

        # Solve numerically
        sol = fsolve(self.ss_value, x0)

        return sol

    def plot_phase_diagram(self, range=1000):
        alpha = self.sim.alpha
        phi = self.sim.phi
        delta = self.sim.delta
        s_K = self.sim.s_K
        s_H = self.sim.s_H
        g = self.sim.g
        n = self.sim.n

        # We create lambdified functions with updated values
        def human_capital_nullcline(kapitaltilde_t):
            return (((delta + g * n + g + n) / s_K) ** (1 / phi)) * kapitaltilde_t ** ((1 - alpha) / phi)

        def physical_capital_nullcline(kapitaltilde_t):
            return (s_H / (delta + g * n + g + n)) ** (1 / (1 - phi)) * kapitaltilde_t ** (alpha / (1 - phi))

        # We then evaluate functions
        tildek_variable = np.linspace(0, range-1, range)
        human_capital_vals = human_capital_nullcline(tildek_variable)
        physical_capital_vals = physical_capital_nullcline(tildek_variable)

        # We create the plot
        plt.plot(human_capital_vals, label="Δh_t=0")
        plt.plot(physical_capital_vals, label="Δk_t=0")
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel('Physical capital')
        plt.ylabel('Human capital')
        plt.title('Phase Diagram')

        # Calculate and display steady state
        ss_values = self.solve_numerical()
        plt.plot(ss_values[0], ss_values[1], 'ro', label='Steady State')
        print("Steady state values (k, h):", ss_values)

        plt.legend()
        plt.show()

    def interactive_NClines(self, range=1000):
        # Create sliders
        capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=self.sim.s_K, description='s_K')
        human_capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=self.sim.s_H, description='s_H')

        # Define interactive function
        def plot_phase_diagram(s_K, s_H):
            alpha = self.sim.alpha
            phi = self.sim.phi
            delta = self.sim.delta
            g = self.sim.g
            n = self.sim.n

            # Define lambdified functions with updated values
            def human_capital_nullcline(kapitaltilde_t):
                return (((delta + g * n + g + n) / s_K) ** (1 / phi)) * kapitaltilde_t ** ((1 - alpha) / phi)

            def physical_capital_nullcline(kapitaltilde_t):
                return (s_H / (delta + g * n + g + n)) ** (1 / (1 - phi)) * kapitaltilde_t ** (alpha / (1 - phi))

            # Evaluate functions
            tildek_variable = np.linspace(0, range-1, range)
            human_capital_vals = human_capital_nullcline(tildek_variable)
            physical_capital_vals = physical_capital_nullcline(tildek_variable)

            # Create plot
            plt.plot(human_capital_vals, label="Δh_t=0")
            plt.plot(physical_capital_vals, label="Δk_t=0")
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.xlabel('Physical capital')
            plt.ylabel('Human capital')
            plt.title('Phase Diagram')

            # Calculate and display steady state
            ss_values = self.solve_numerical()
            plt.plot(ss_values[0], ss_values[1], 'ro', label='Steady State')
            plt.text(ss_values[0] + 1, ss_values[1], f'Steady State: ({ss_values[0]:.2f}, {ss_values[1]:.2f})', fontsize=9)

            plt.legend()
            plt.show()

        # Call interactive function
        widgets.interact(plot_phase_diagram, s_K=capital_savings_rate_slider, s_H=human_capital_savings_rate_slider)