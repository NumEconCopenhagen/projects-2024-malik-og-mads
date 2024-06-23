import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from types import SimpleNamespace
from ipywidgets import interact, FloatSlider

class Equation:
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
    
class Solowvalue:
    def __init__(self):
        # Setting up parameters and simulation values using dictionaries
        self.par = {'alpha': 0.3, 'n': 0.02, 'g': 0.01, 's': 0.2, 'delta': 0.05}
        self.num_periods = 99
        self.k_0 = 1e-7

    def ss_equation(self):
        par = self.par
        # Closed form solution derived from the steady state equation without sympy
        ss = ((par['s'] / (par['g'] + par['n'] + par['delta'])) ** (1 / (1 - par['alpha'])))
        return ss
    
class trans_dia:
    def __init__(self, alpha=0.3, n=0.02, g=0.01, s=0.2, delta=0.05, k_0=1e-7, num_periods=99):
        self.alpha = alpha
        self.n = n
        self.g = g
        self.s = s
        self.delta = delta
        self.k_0 = k_0
        self.num_periods = num_periods

    def transition_diagram(self):
        k_t = self.k_0
        capital_values = [k_t]

        for _ in range(self.num_periods):
            k_t1 = (self.s * (k_t ** self.alpha) + (1 - self.delta) * k_t) / (1 + self.g + self.n)
            capital_values.append(k_t1)
            k_t = k_t1

        # Plot the transitions
        plt.plot(capital_values[:-1], capital_values[1:], 'b-', linewidth=1.5, label='k_t+1')
        plt.plot(capital_values[:-1], capital_values[:-1], 'r-', linewidth=1.5, label='k_t = k_t+1')
        plt.xlabel('Capital level at t')
        plt.ylabel('Capital level at t+1')
        plt.title('Transition diagram of capital')
        plt.legend()
        plt.grid(True)
        plt.show()
    
class inter_trans_dia:
    def __init__(self, alpha=0.3, n=0.02, g=0.01, s=0.2, delta=0.05, k_0=1e-7, num_periods=99):
        self.alpha = alpha
        self.n = n
        self.g = g
        self.s = s
        self.delta = delta
        self.k_0 = k_0
        self.num_periods = num_periods

    def set_savings_rate(self, s):
        self.s = s

    def transition_diagram(self):
        import matplotlib.pyplot as plt
        k_t = self.k_0
        capital_values = [k_t]

        for _ in range(self.num_periods):
            k_t1 = (self.s * (k_t ** self.alpha) + (1 - self.delta) * k_t) / (1 + self.g + self.n)
            capital_values.append(k_t1)
            k_t = k_t1

        plt.figure(figsize=(10, 6))
        plt.plot(capital_values[:-1], capital_values[1:], 'b-', linewidth=1.5, label='k_t+1')
        plt.plot(capital_values[:-1], capital_values[:-1], 'r-', linewidth=1.5, label='k_t = k_t+1')
        plt.xlabel('Capital level at t')
        plt.ylabel('Capital level at t+1')
        plt.title('Transition diagram of capital')
        plt.legend()
        plt.grid(True)
        plt.show()

class Solow2_equations:
    def __init__(self):
        # Setting up parameter namespaces
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # Defining parameters
        sim.n = 0.02
        sim.delta = 0.05
        sim.g = 0.01
        sim.s_H = 0.15
        sim.phi = 0.3
        sim.s_K = 0.2
        sim.alpha = 0.3

        # Giving our paramters and variables names
        par.n, par.delta, par.g, par.s_H, par.phi, par.s_K, par.alpha, par.A_t, par.L_t, par.Y_t, par.K_t, par.H_t = sm.symbols('n delta g s_H phi s_K alpha A_t L_t Y_t K_t H_t')

        # Naming per effective worker variables
        par.kapitaltilde_t = sm.symbols('kt')
        par.humantilde_t = sm.symbols('ht')

    def ss_value_k(k, h, alpha, delta, s_K, s_H, g, n, phi, do_print=False):
        # Symbolic variables initialization:
        g, k, h, alpha, s_K, s_H, n, delta, phi = sm.symbols('g k h alpha s_K s_H n delta phi')
        
        # Steady state for human capital:
        steadystate_h = sm.Eq(h, 1/((1+g) * (1+n)) * ((1-delta)*h + (s_H) * h**phi * k**alpha))
        hsteadystate = sm.solve(steadystate_h, h)[0]

        # Steady state value for physical capital: 
        steadystate_k = sm.Eq(k, 1/((1+g) * (1+n)) * ((1 - delta)*k + (s_K) * h**phi * k**alpha))
        ksteadystate = sm.solve(steadystate_k, k)[0]
        
        # Substituting and solving for steady state physical capital: Substitutes the steady state value of human capital into the physical capital equation and solves it.
        ss_physicalcapital = ksteadystate.subs(h, hsteadystate)
        return ss_physicalcapital

    def ss_value_h(k, h, alpha, delta, s_K, s_H, g, n, phi, do_print=False):
        # Symbolic variables initialization:
        g, k, h, alpha, s_K, s_H, n, delta, phi = sm.symbols('g k h alpha s_K s_H n delta phi')

        # Steady state for human capital: 
        steadystate_h = sm.Eq(h, 1/((1+g) * (1+n)) * ((1-delta)*h + (s_H) * h**phi * k**alpha))
        hsteadystate = sm.solve(steadystate_h, h)[0]

        # Steady state value for physical capital: 
        steadystate_k = sm.Eq(k, 1/((1+g) * (1+n)) * ((1 - delta)*k + (s_K) * h**phi * k**alpha))
        ksteadystate = sm.solve(steadystate_k, k)[0]

        # Substituting and solving for steady state human capital : Uses the solved value of physical capital to find the steady state human capital.
        ss_humancapital = hsteadystate.subs(k, ksteadystate)
        return ss_humancapital
    
# Define the steady state equations as Python functions
def h_steady_state(alpha, phi, delta, n, g, s_K, s_H):
    """Calculate the steady state for human capital."""
    tildeh = ((s_H**(1 - alpha) * s_K**alpha) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
    return tildeh

def k_steady_state(alpha, phi, delta, n, g, s_K, s_H):
    """Calculate the steady state for physical capital."""
    tildek = ((s_H**phi * s_K**(1 - phi)) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
    return tildek


class trans_dia2:
    def __init__(self, alpha=0.3, phi=0.3, delta=0.05, n=0.02, g=0.01, s_K=0.2, s_H=0.15):
        self.alpha = alpha
        self.phi = phi
        self.delta = delta
        self.n = n
        self.g = g
        self.s_K = s_K
        self.s_H = s_H

    def calculate_nullcline(self, kapitaltilde_t, s_K, s_H):
        alpha = self.alpha
        phi = self.phi
        delta = self.delta
        n = self.n
        g = self.g
        # Human capital nullcline calculation
        human_capital_nullcline = (((delta + g * n + g + n) / s_K) ** (1 / phi)) * kapitaltilde_t ** ((1 - alpha) / phi)
        # Physical capital nullcline calculation
        physical_capital_nullcline = (s_H / (delta + g * n + g + n)) ** (1 / (1 - phi)) * kapitaltilde_t ** (alpha / (1 - phi))
        return human_capital_nullcline, physical_capital_nullcline

    def NClines(self):
        def calculate_steady_state(s_K, s_H, delta, n, g, alpha, phi):
            # Calculate steady states for human and physical capital
            h_ss = ((s_H**(1 - alpha) * s_K**alpha) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
            k_ss = ((s_H**phi * s_K**(1 - phi)) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
            return k_ss, h_ss

        # Constants from the model
        alpha = self.alpha
        phi = self.phi
        delta = self.delta
        n = self.n
        g = self.g
        s_K = self.s_K
        s_H = self.s_H

        # Generate the range and calculate nullclines
        tildek_variable = np.linspace(0, 20, 400)
        human_capital_vals, physical_capital_vals = self.calculate_nullcline(tildek_variable, s_K, s_H)

        # Calculate the steady state
        k_ss, h_ss = calculate_steady_state(s_K, s_H, delta, n, g, alpha, phi)

        # Plotting the phase diagram
        plt.figure(figsize=(8, 6))
        plt.plot(tildek_variable, human_capital_vals, label="Δh_t=0 (Human Capital Nullcline)")
        plt.plot(tildek_variable, physical_capital_vals, label="Δk_t=0 (Physical Capital Nullcline)")
        plt.scatter(k_ss, h_ss, color='red', zorder=5, label='Steady State')  # Plot the steady state as a red dot
        plt.xlabel('Physical capital per effective worker')
        plt.ylabel('Human capital per effective worker')
        plt.title('Phase Diagram')
        plt.legend()
        plt.grid(True)
        plt.show()


    def interactive_NClines(self):
        def calculate_steady_state(s_K, s_H, delta, n, g, alpha, phi):
            # Calculate steady states for human and physical capital
            h_ss = ((s_H**(1 - alpha) * s_K**alpha) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
            k_ss = ((s_H**phi * s_K**(1 - phi)) / (delta + g * n + g + n))**(1 / (1 - alpha - phi))
            return k_ss, h_ss

        def plot_interactive(s_K, s_H):
            alpha = self.alpha
            phi = self.phi
            delta = self.delta
            n = self.n
            g = self.g

            # Generate the range and calculate nullclines
            tildek_variable = np.linspace(0, 20, 400)
            human_capital_vals, physical_capital_vals = self.calculate_nullcline(tildek_variable, s_K, s_H)

            # Calculate the steady state
            k_ss, h_ss = calculate_steady_state(s_K, s_H, delta, n, g, alpha, phi)

            # Plotting the phase diagram
            plt.figure(figsize=(8, 6))
            plt.plot(tildek_variable, human_capital_vals, label="Δh_t=0 (Human Capital Nullcline)")
            plt.plot(tildek_variable, physical_capital_vals, label="Δk_t=0 (Physical Capital Nullcline)")
            plt.scatter(k_ss, h_ss, color='red', zorder=5, label='Steady State')  # Plot the steady state as a red dot
            plt.xlabel('Physical capital per effective worker')
            plt.ylabel('Human capital per effective worker')
            plt.title('Phase Diagram')
            plt.legend()
            plt.grid(True)
            plt.show()

        s_K_slider = FloatSlider(min=0.01, max=0.5, step=0.01, value=self.s_K, description='s_K', continuous_update=False)
        s_H_slider = FloatSlider(min=0.01, max=0.5, step=0.01, value=self.s_H, description='s_H', continuous_update=False)
        interact(plot_interactive, s_K=s_K_slider, s_H=s_H_slider)