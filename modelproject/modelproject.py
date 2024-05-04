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
        transition = sm.Eq(par.k, 1/((1+par.n)*(1+par.g))*((1-par.delta)*par.k + par.s*f))

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
            k_t1 = 1 / ((1 + self.sim.n) * (1 + self.sim.g)) * (
                    self.sim.s * k_t ** self.sim.alpha + (1 - self.sim.delta) * k_t)
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
        s_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.2, description='s')

        # Interactive update of variable s
        def update_s_slider(s):
            self.sim.s = s
            self.transition_diagram()

        interactive_plot = widgets.interactive(update_s_slider, s=s_slider)
        display(interactive_plot)



class Solow_H:
    def __init__(self):
        # Defining namespaces for parameters
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
        par.s_H = sm.symbols('s_{H}')
        par.phi = sm.symbols('phi')
        par.n = sm.symbols('n')
        par.g = sm.symbols('g')
        par.delta = sm.symbols('delta')
        par.s_K = sm.symbols('s_{K}')

        # Naming variables
        par.A_t = sm.symbols('A_{t}')
        par.K_t = sm.symbols('K_{t}')
        par.Y_t = sm.symbols('Y_{t}')
        par.H_t = sm.symbols('H_{t}')
        par.L_t = sm.symbols('L_{t}')

        # Naming per effective worker variables
        par.ktilde_t = sm.symbols('\tilde{k_{t}}')
        par.htilde_t = sm.symbols('\tilde{h_{t}}')

    def ss_value_k(k,h,alpha,delta,s_K,s_H,g,n,phi, do_print=False):
        # Symbolic variables
        g = sm.symbols('g')
        h = sm.symbols('h')
        alpha = sm.symbols('alpha')
        s_K = sm.symbols('s_K')
        s_H = sm.symbols('s_H')
        k = sm.symbols('k')
        n = sm.symbols('n')
        delta = sm.symbols('delta')
        phi = sm.symbols('phi')
        y = k**alpha * h**phi
        
        # Steady state value function
        ss_k = sm.Eq(k, 1/((1+n)*(1+g))*((s_K)*y+(1-delta)*k))
        kss = sm.solve(ss_k,k)[0]
        
        # Steady state for human capital
        ss_h = sm.Eq(h, 1/((1+n)*(1+g)) * ((s_H)*y+(1-delta)*h) )
        hss = sm.solve(ss_h,h)[0]

        # Substituting and solving
        k_ss = kss.subs(h,hss)
        h_ss = hss.subs(k,kss)
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
        y = k**alpha * h**phi

        # Steady state value function
        ss_k = sm.Eq(k, 1/((1+n)*(1+g))*((s_K)*y+(1-delta)*k))
        kss = sm.solve(ss_k,k)[0]

        # Steady state for human capital
        ss_h = sm.Eq(h, 1/((1+n)*(1+g)) * ((s_H)*y+(1-delta)*h) )
        hss = sm.solve(ss_h,h)[0]

        # Substituting and solving
        k_ss = kss.subs(h,hss)
        h_ss = hss.subs(k,kss)

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

    def Nullclines(self, periods=500):
        par = self.par
        sim = self.sim
        periods = periods

        # Define interactive function
        def plot_function(s_K, s_H, alpha_phi, delta, periods):
            alpha = alpha_phi
            phi = alpha_phi

            # Create lambdified functions with updated values
            ncht_expr = (((par.n + par.g + delta + par.n * par.g) / s_K) ** (1 / phi)) * (par.ktilde_t ** ((1 - alpha) / phi))
            nckt_expr = (s_H/(par.n+par.g+delta+par.n*par.g))**(1/(1-phi))*par.ktilde_t**(alpha/(1-alpha))
            ncht_func = sm.lambdify(par.ktilde_t, ncht_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_K: s_K}))
            nckt_func = sm.lambdify(par.ktilde_t, nckt_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_H: s_H}))

            # Evaluate functions
            ktilde_vals = np.linspace(0, periods-1, periods)
            ncht_vals = ncht_func(ktilde_vals)
            htilde_vals = np.linspace(0, periods-1, periods)
            nckt_vals = nckt_func(htilde_vals)

            # Create plot
            plt.plot(ncht_vals, label="Δh_t=0")
            plt.plot(nckt_vals, label="Δk_t=0")
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.xlabel('Level of physical capital')
            plt.ylabel('Level of human capital')
            plt.title('Phase Diagram')

            # Calculate and display steady state
            try:
                ktilde_expr = ((s_K**(1-phi) * s_H**phi)/(par.n + par.g + delta + par.n*par.g))**(1/(1-phi-alpha))
                htilde_expr = ((s_K**alpha * s_H**(1-alpha))/(par.n + par.g + delta + par.n*par.g))**(1/(1-phi-alpha))
                ktilde_func = sm.lambdify([], [ktilde_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                htilde_func = sm.lambdify([], [htilde_expr.subs({par.phi: phi, par.alpha: alpha, par.delta: delta, par.s_K: s_K, par.s_H: s_H, par.n: sim.n, par.g: sim.g})])
                ktilde_steady_state = ktilde_func()[0]
                htilde_steady_state = htilde_func()[0]
                plt.plot(ktilde_steady_state, htilde_steady_state, 'ro', label='Steady State')
            except Exception as e:
                print(f"Error calculating steady state: {e}")

            # Display legend
            plt.legend()

            # Show plot
            plt.show()

        # Create sliders
        s_K_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.2, description='s_K')
        s_H_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.15, description='s_H')
        alpha_phi_slider = widgets.FloatSlider(min=0.001, max=0.5, step=0.001, value=0.3, description='alpha/phi')
        delta_slider = widgets.FloatSlider(min=0.001, max=0.1, step=0.001, value=0.05, description='delta')
        periods_dropdown = widgets.Dropdown(options=list(range(100, 1001, 100)), value=100, description='periods')

        # Call interactive function
        widgets.interact(plot_function, s_K=s_K_slider, s_H=s_H_slider, alpha_phi=alpha_phi_slider, delta=delta_slider, periods=periods_dropdown)
