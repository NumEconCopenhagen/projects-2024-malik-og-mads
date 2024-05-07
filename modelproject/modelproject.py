# Importing necessary libraries
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from IPython.display import display
import ipywidgets as widgets
import sympy as sm


class Solow:
    def __init__(self):

        # Setting up parameters and namespaces
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # Defining symbolic parameters
        par.K, par.A, par.L, par.Y, par.S = sm.symbols('K A L Y S')
        par.alpha, par.delta, par.s, par.g, par.n = sm.symbols('alpha delta s g n')

        # Defining capital per effective worker
        par.k = sm.symbols('k')

        # Defining parameters
        sim.alpha = 0.3
        sim.delta = 0.05
        sim.s = 0.2  
        sim.g = 0.01
        sim.n = 0.02
        self.num_periods = 100
        self.k_0 = 1e-7

    def ss_equation(self):
        par = self.par
        # Defining the transition equation
        f = par.k**par.alpha
        transition = sm.Eq(par.k, 1/((1+par.g)*(1+par.n))*(par.s*f + (1-par.delta)*par.k))

        # Solving for the steady state
        steady_state = sm.solve(transition, par.k)[0]

        return steady_state
    
    def ss_value(self): 

        par = self.par
        sim = self.sim

        # Converting symbolic steady state into a function
        ss_function = sm.lambdify((par.s,par.g,par.n,par.delta,par.alpha), self.ss_equation())

        return ss_function(sim.s,sim.g,sim.n,sim.delta,sim.alpha)
    
    def transition_diagram(self):

        capital_values = [self.k_0]  # List to store the capital stock values

        # Calculate the capital stock values for each period
        for _ in range(self.num_periods):
            k_t = capital_values[-1]  # Current capital stock
            k_t1 = 1 / ((1 + self.sim.g)* (1 + self.sim.n)) * (
                    (1 - self.sim.delta) * k_t + self.sim.s * k_t ** self.sim.alpha)
            capital_values.append(k_t1)

        # Plotting the transition diagram
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
        # FloatSlider for the variable s
        s_slider = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.2, description='s')

        # Interactive update of variable s
        def update_s_slider(s):
            self.sim.s = s
            self.transition_diagram()

        interactive_plot = widgets.interactive(update_s_slider, s=s_slider)
        display(interactive_plot)

class Solow_H:
    def __init__(self):
        # Setting up parameter namespaces
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # Defining parameters
        sim.alpha = 0.3
        sim.phi = 0.3
        sim.delta = 0.05
        sim.n = 0.02
        sim.g = 0.01
        sim.s_K = 0.2
        sim.s_H = 0.15

        # Naming the parameters
        par.alpha = sm.symbols('alpha')
        par.s_H = sm.symbols('s_H')
        par.phi = sm.symbols('phi')
        par.n = sm.symbols('n')
        par.g = sm.symbols('g')
        par.delta = sm.symbols('delta')
        par.s_K = sm.symbols('s_K')

        # Naming variables
        par.Y_t = sm.symbols('Y_t')
        par.A_t = sm.symbols('A_t')
        par.H_t = sm.symbols('H_t')
        par.L_t = sm.symbols('L_t')
        par.K_t = sm.symbols('K_t')

        # Naming per effective worker variables
        par.kapitaltilde_t = sm.symbols('tk')
        par.humantilde_t = sm.symbols('th')

    def ss_value_k(k,h,alpha,delta,s_K,s_H,g,n,phi, do_print=False):
        # Symbolic variables
        g = sm.symbols('g')
        k = sm.symbols('k')
        h = sm.symbols('h')
        alpha = sm.symbols('alpha')
        s_K = sm.symbols('s_K')
        s_H = sm.symbols('s_H')
        n = sm.symbols('n')
        delta = sm.symbols('delta')
        phi = sm.symbols('phi')
        
         # Steady state for human capital
        steadystate_h = sm.Eq(h, 1/((1+g) * (1+n)) * ((1-delta)*h + (s_H) * h**phi * k**alpha))
        hsteadystate = sm.solve(steadystate_h,h)[0]

        # Steady state value function
        steadystate_k = sm.Eq(k, 1/((1+g) * (1+n)) * ((1 - delta)*k + (s_K) * h**phi * k**alpha))
        ksteadystate = sm.solve(steadystate_k,k)[0]
        

        # Substituting and solving
        k_ss = ksteadystate.subs(h,hsteadystate)
        return k_ss

    def ss_value_h(k,h,alpha,delta,s_K,s_H,g,n,phi, do_print=False):
        # Symbolic variables
        k = sm.symbols('k')
        delta = sm.symbols('delta')
        phi = sm.symbols('phi')
        h = sm.symbols('h')
        alpha = sm.symbols('alpha')
        g = sm.symbols('g')
        s_K = sm.symbols('s_K')
        s_H = sm.symbols('s_H')
        n = sm.symbols('n')

        # Steady state for human capital
        steadystate_h = sm.Eq(h, 1/((1+g) * (1+n)) * ((1-delta)*h + (s_H) * h**phi * k**alpha))
        hsteadystate = sm.solve(steadystate_h,h)[0]

        # Steady state value function
        steadystate_k = sm.Eq(k, 1/((1+g) * (1+n)) * ((1 - delta)*k + (s_K) * h**phi * k**alpha))
        ksteadystate = sm.solve(steadystate_k,k)[0]


        # Substituting and solving
        h_ss = hsteadystate.subs(k,ksteadystate)

        return h_ss

    def ss_functions(self,alpha,phi,delta,n,g,s_K,s_H):
        par = self.sim
        alpha = par.alpha
        phi = par.phi
        delta = par.delta
        n = sim.n
        g= sim.g
        s_K = sim.s_K
        s_H = sim.s_H

        # Steady state functions
        k_tilde = ((s_K**(1-phi) * s_H**phi)/(n+g+delta +n*g))**(1/(1-phi-alpha))
        h_tilde = ( (s_K**(alpha) * s_H**(1-alpha))/(n+g+delta +n*g))**(1/(1-phi-alpha))

        # Lambdify for Python function
        kss_function = sm.lambdify((alpha,phi,delta,n,g,s_K,s_H),k_tilde)
        hss_function = sm.lambdify((alpha,phi,delta,n,g,s_K,s_H),h_tilde) 

        # Calculate SS values
        kss_function(alpha,phi,delta,n,g,s_K,s_H)
        hss_function(alpha,phi,delta,n,g,s_K,s_H)

        return 'SS value for k: ', kss_function(alpha,phi,delta,n,g,s_K,s_H) ,'and ss value for h:',  hss_function(alpha,phi,delta,n,g,s_K,s_H)


    def NClines(self, range=1000, delta=0.05, productivity_growth_params=0.3):
        par = self.par
        sim = self.sim

        # Define plotting function with fixed savings rates
        def plot_phase_diagram(s_K=0.2, s_H=0.15):
            alpha = productivity_growth_params
            phi = productivity_growth_params

            # Create lambdified functions with updated values
            human_capital_nullcline_expr = (((delta + par.g * par.n + par.g + par.n) / s_K) ** (1 / phi)) * par.kapitaltilde_t ** ((1 - alpha) / phi)
            physical_capital_nullcline_expr = (s_H / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - phi)) * par.kapitaltilde_t ** (alpha / (1 - phi))
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
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.xlabel('Physical capital')
            plt.ylabel('Human capital')
            plt.title('Phase Diagram')

            # Calculate and display steady state
            try:
                htilde_steady_state_expr = ((s_H ** (1 - alpha) * s_K ** alpha) / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - alpha - phi))
                ktilde_steady_state_expr = ((s_H ** phi * s_K ** (1 - phi)) / (delta + par.g * par.n + par.g + par.n)) ** (1 / (1 - alpha - phi))
                htilde_steady_state_func = sm.lambdify([], [htilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                ktilde_steady_state_func = sm.lambdify([], [ktilde_steady_state_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                tildeh_ss = htilde_steady_state_func()[0]
                tildek_ss = ktilde_steady_state_func()[0]
                plt.plot(tildek_ss, tildeh_ss, 'ro', label='Steady State')
            except Exception as e:
                print(f"Error calculating steady state: {e}")

            # Display legend
            plt.legend()
            # Show plot
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
                print(f"Error calculating steady state: {e}")

            # Display legend
            plt.legend()
            # Show plot
            plt.show()

        # Create sliders
        capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.2, description='s_K')
        human_capital_savings_rate_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.15, description='s_H')

        # Call interactive function
        widgets.interact(plot_phase_diagram, s_K=capital_savings_rate_slider, s_H=human_capital_savings_rate_slider)


