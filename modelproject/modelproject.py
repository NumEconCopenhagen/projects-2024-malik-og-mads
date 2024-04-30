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
    