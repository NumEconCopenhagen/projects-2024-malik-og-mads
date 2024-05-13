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

    def NClines(self, range=1000, delta=0.05, productivity_growth_params=0.3):
        par = self.par
        sim = self.sim

        # Define plotting function with fixed savings rates
        def plot_phase_diagram(s_K=0.2, s_H=0.15):
            alpha = productivity_growth_params
            phi = productivity_growth_params

            # We create lambdified functions with updated values
            human_capital_nullcline_expr = (((delta + par.g * par.n + par.g + par.n) / s_K) ** (1 / phi)) * par.kapitaltilde_t ** ((1 - alpha) / phi)
            physical_capital_nullcline_expr = (s_H / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - phi)) * par.kapitaltilde_t ** (alpha / (1 - phi))
            human_capital_nullcline_func = sm.lambdify(par.kapitaltilde_t, human_capital_nullcline_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_K: s_K}))
            physical_capital_nullcline_func = sm.lambdify(par.kapitaltilde_t, physical_capital_nullcline_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_H: s_H}))

            # We then evaluate functions
            tildek_variable = np.linspace(0, range-1, range)
            human_capital_vals = human_capital_nullcline_func(tildek_variable)
            tildeh_variable = np.linspace(0, range-1, range)
            physical_capital_vals = physical_capital_nullcline_func(tildeh_variable)

            # We create the plot
            plt.plot(human_capital_vals, label="Δh_t=0")
            plt.plot(physical_capital_vals, label="Δk_t=0")
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.xlabel('Physical capital')
            plt.ylabel('Human capital')
            plt.title('Phase Diagram')

            # We calculate and display ss
            try:
                htilde_steady_state_expr = ((s_H ** (1 - alpha) * s_K ** alpha) / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - alpha - phi))
                ktilde_steady_state_expr = ((s_H ** phi * s_K ** (1 - phi)) / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - alpha - phi))
                htilde_steady_state_func = sm.lambdify([], [htilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                ktilde_steady_state_func = sm.lambdify([], [ktilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                tildeh_ss = htilde_steady_state_func()[0]
                tildek_ss = ktilde_steady_state_func()[0]
                plt.plot(tildek_ss, tildeh_ss, 'ro', label='Steady State')
            except Exception as e:
                print(f"Error: {e}")

            plt.legend()
            plt.show()

        # Call plotting function directly
        plot_phase_diagram()


    def interactive_NClines(self, range=1000, delta=0.05, productivity_growth_params=0.3):
        par = self.par
        sim = self.sim

        # Define interactive function
        def plot_phase_diagram(s_K, s_H):
            alpha = productivity_growth_params
            phi = productivity_growth_params

            # Create lambdified functions with updated values
            human_capital_nullcline_expr = (((delta + par.g * par.n + par.g + par.n) / s_K) ** (1 / phi)) * par.kapitaltilde_t ** ((1 - alpha) / phi)
            physical_capital_nullcline_expr = (s_H/(delta + par.g * par.n + par.g + par.n))**(1/(1-phi)) * par.kapitaltilde_t**(alpha/(1-phi))
            human_capital_nullcline_func = sm.lambdify(par.kapitaltilde_t, human_capital_nullcline_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_K: s_K}))
            physical_capital_nullcline_func = sm.lambdify(par.kapitaltilde_t, physical_capital_nullcline_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_H: s_H}))

            # Evaluate functions
            tildek_variable = np.linspace(0, range-1, range)
            human_capital_vals = human_capital_nullcline_func(tildek_variable)
            tildeh_variable = np.linspace(0, range-1, range)
            physical_capital_vals = physical_capital_nullcline_func(tildeh_variable)

            # Create plot
            plt.plot(human_capital_vals, label="Δh_t=0")
            plt.plot(physical_capital_vals, label="Δk_t=0")
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.xlabel('Physical capital')
            plt.ylabel('Human capital')
            plt.title('Phase Diagram')

            # Calculate and display steady state
            try:
                htilde_steady_state_expr = ((s_H**(1-alpha) * s_K**alpha)/(delta + par.g * par.n + par.g + par.n))**(1/(1-alpha-phi))
                ktilde_steady_state_expr = ((s_H**phi * s_K**(1-phi))/(delta + par.g * par.n + par.g + par.n))**(1/(1-alpha-phi))
                htilde_steady_state_func = sm.lambdify([], [htilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                ktilde_steady_state_func = sm.lambdify([], [ktilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                tildeh_ss  = htilde_steady_state_func()[0]
                tildek_ss  = ktilde_steady_state_func()[0]
                plt.plot(tildek_ss , tildeh_ss , 'ro', label='Steady State')
            except Exception as e:
                print(f"Error: {e}")

            plt.legend()
            plt.show()

        # Create sliders
        capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.2, description='s_K')
        human_capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.15, description='s_H')

        # Call interactive function
        widgets.interact(plot_phase_diagram, s_K=capital_savings_rate_slider, s_H=human_capital_savings_rate_slider)


