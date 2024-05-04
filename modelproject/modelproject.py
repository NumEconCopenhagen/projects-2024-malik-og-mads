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
        sim.g = 0.01
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
        # FloatSlider for the variable s
        s_slider = widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.2, description='s')

        # Interactive update of variable s
        def update_s_slider(s):
            self.sim.s = s
            self.transition_diagram()

        interactive_plot = widgets.interactive(update_s_slider, s=s_slider)
        display(interactive_plot)


# Creating a class for the solowmodel with human capital
class Solow_H:
    def __init__(self):
                # We start making namespaces: 
        par = self.par = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # We define our sim parameters, these will create the ground for our simulation.
        sim.alpha = 0.3
        sim.phi = 0.3
        sim.delta = 0.05
        sim.n = 0.02
        sim.g = 0.01
        sim.s_K = 0.2
        sim.s_H = 0.15

        # We name the parameters:
        par.alpha = sm.symbols('alpha')
        par.phi  = sm.symbols('phi') 
        par.delta = sm.symbols('delta') 
        par.n = sm.symbols('n') 
        par.g = sm.symbols('g') 
        par.s_K = sm.symbols('s_{K}') 
        par.s_H = sm.symbols('s_{H}') 

        # We name variables
        par.K_t = sm.symbols('K_{t}') 
        par.H_t = sm.symbols('H_{t}')
        par.Y_t = sm.symbols('Y_{t}')
        par.A_t = sm.symbols('A_{t}')
        par.L_t = sm.symbols('L_{t}')
        
        # # We name our per effective worker variables
        par.ktilde_t = sm.symbols('\tilde{k_{t}}')
        par.htilde_t = sm.symbols('\tilde{h_{t}}')

    def ss_value_k(k,h,alpha,delta,s_K,s_H,g,n,phi, do_print=False):
        k = sm.symbols('k')
        h = sm.symbols('h')
        alpha = sm.symbols('alpha')
        delta = sm.symbols('delta')
        s_K = sm.symbols('s_K')
        s_H = sm.symbols('s_H')
        g = sm.symbols('g')
        n = sm.symbols('n')
        phi = sm.symbols('phi')
        y = k**alpha * h**phi
        # We define the function for the ss value 
        ss_k = sm.Eq(k, 1/((1+n)*(1+g))*((s_K)*y+(1-delta)*k)) 
        # We find ss value for k. Here we put left hand side equal to zero and solve
        kss = sm.solve(ss_k,k)[0]
                
        # We do it for human capital
        ss_h = sm.Eq(h, 1/((1+n)*(1+g)) * ((s_H)*y+(1-delta)*h) ) 
        hss = sm.solve(ss_h,h)[0]

        # We substitute h in kss and solve for k
        k_ss = kss.subs(h,hss)
        # Here we stubsitute k in hss and solve h
        h_ss = hss.subs(k,kss)
        return k_ss

    def ss_value_h(k,h,alpha,delta,s_K,s_H,g,n,phi, do_print=False):
        k = sm.symbols('k')
        h = sm.symbols('h')
        alpha = sm.symbols('alpha')
        delta = sm.symbols('delta')
        s_K = sm.symbols('s_K')
        s_H = sm.symbols('s_H')
        g = sm.symbols('g')
        n = sm.symbols('n')
        phi = sm.symbols('phi')
        y = k**alpha * h**phi

        # function for ss value
        ss_k = sm.Eq(k, 1/((1+n)*(1+g))*((s_K)*y+(1-delta)*k)) 
        # Again we find ss value by letting left hand side equal to zero
        kss = sm.solve(ss_k,k)[0]
                
        # The same as before but for h
        ss_h = sm.Eq(h, 1/((1+n)*(1+g)) * ((s_H)*y+(1-delta)*h) ) 
        hss = sm.solve(ss_h,h)[0]

        # substituting
        k_ss = kss.subs(h,hss)

        # and substituting again
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


        # Here we is the ss functions
        k_tilde = ((s_K**(1-phi) * s_H**phi)/(n+g+delta +n*g))**(1/(1-phi-alpha))
        h_tilde = ( (s_K**(alpha) * s_H**(1-alpha))/(n+g+delta +n*g))**(1/(1-phi-alpha))
        
        # Lambdify is used to use the python function
        kss_function = sm.lambdify((alpha,phi,delta,n,g,s_K,s_H),k_tilde)
        hss_function = sm.lambdify((alpha,phi,delta,n,g,s_K,s_H),h_tilde) 

        # SS values are calculated
        kss_function(alpha,phi,delta,n,g,s_K,s_H)
        hss_function(alpha,phi,delta,n,g,s_K,s_H)

        return 'SS value for k: ', kss_function(alpha,phi,delta,n,g,s_K,s_H) ,'and ss value for h:',  hss_function(alpha,phi,delta,n,g,s_K,s_H)

    # We now define our method for simulating the Nullclines for our extended model
    def Nullclines(self, periods=500):
        par = self.par
        sim = self.sim
        periods = periods

        # Define the interactive function
        def plot_function(s_K, s_H, alpha_phi, delta, periods):
            alpha = alpha_phi
            phi = alpha_phi

            # Create the lambdified functions with updated s_K, s_H, alpha, phi, and delta values
            ncht_expr = (((par.n + par.g + delta + par.n * par.g) / s_K) ** (1 / phi)) * (par.ktilde_t ** ((1 - alpha) / phi))
            nckt_expr = (s_H/(par.n+par.g+delta+par.n*par.g))**(1/(1-phi))*par.ktilde_t**(alpha/(1-alpha))
            ncht_func = sm.lambdify(par.ktilde_t, ncht_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_K: s_K}))
            nckt_func = sm.lambdify(par.ktilde_t, nckt_expr.subs({par.alpha: alpha, par.phi: phi, par.delta: delta, par.n: sim.n, par.g: sim.g, par.s_H: s_H}))

            # Evaluate the functions for different t_values
            ktilde_vals = np.linspace(0, periods-1, periods)
            ncht_vals = ncht_func(ktilde_vals)
            htilde_vals = np.linspace(0, periods-1, periods)
            nckt_vals = nckt_func(htilde_vals)

            # Create the plot
                       
            plt.plot(ncht_vals, label="Δh_t=0")
            plt.plot(nckt_vals, label="Δk_t=0")
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.xlabel('Level of physical capital')
            plt.ylabel('Level of human capital')
            plt.title('Phasediagram')

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

            # Display the legend
            
            plt.legend()

            # Show the plot
            
            plt.show()

        # Create FloatSliders for s_K, s_H, alpha_phi, delta, and periods
        s_K_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.2, description='s_K')
        s_H_slider = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.15, description='s_H')
        alpha_phi_slider = widgets.FloatSlider(min=0.001, max=0.5, step=0.001, value=0.3, description='alpha/phi')
        delta_slider = widgets.FloatSlider(min=0.001, max=0.1, step=0.001, value=0.05, description='delta')
        periods_dropdown = widgets.Dropdown(options=list(range(100, 1001, 100)), value=100, description='periods')

        # Call the interactive function with the sliders as arguments
        widgets.interact(plot_function, s_K=s_K_slider, s_H=s_H_slider, alpha_phi=alpha_phi_slider, delta=delta_slider, periods=periods_dropdown)
