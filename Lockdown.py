# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:25:46 2020

@author: james
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:16:42 2020

@author: james
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

## Set up some initial conditions

# Population of size 1, i.e. 100% (N is between 0 and 1)
N = 1

# Assume some Infected people (1% are Infected)
Istart = 0.01

# Assume some people are Susceptible
Sstart = N - Istart

# Nobody yet has Recovered
Rstart = 0

#print(f"Starting conditions: N = {N}, S = {Sstart}, I = {Istart}, R = {Rstart}")

## For now, fix these "rate" variables

# transm = Transmission/infection rate, how quickly the disease gets transmitted.
transm = 3.2

# recov = Recovery rate, how quickly people recover, this should be
# smaller as it takes people longer to recover from a disease.
death = 0.14417
recov = 1 - death

# maxT = How long we're going to let the model run for.
maxT = 0.5

## Let's write these in Python.


def dS_dT(S, I, transm, recov):
    """The rate of change of Susceptibles over time.
    
    Args:
        S (float): Total who are Susceptible.
        I (float): Total who are Infected.
        transm (float): transmission rate.
    
    Returns:
        float: rate of change of Suscpetibles.
    
    Examples:
        
        >> dS_dT(S=0.99, I=0.01, transm=3.2)
        -0.03168
    """
    # Negative because rate will go down as more Susceptible people get Infected.
    return -transm * S * I + recov * I


def dI_dT(S, I, transm, recov, death):
    """The rate of change of Infected people over time.
    
    Args:
        S (float): Total who are Susceptible.
        I (float): Total who are Infected.
        transm (float): transmission rate.
        recov (float): recovery rate.
    
    Returns:
        float: rate of change of Infected.
    
    Examples:
    
        >> dI_dT(S=0.99, I=0.01, transm=3.2, recov=0.23)
        0.02938
    """
    return (
        transm * S * I  # If people were Susceptible, they'll become Infected next.
        - recov * I  # The more people become Infected, the more people can Recover.
        - death * I
    )


def dD_dT(I, death):
    """The rate of change of Recovered people over time.
    
    Args:
        I (float): Total who are Infected.
        recov (float): recovery rate.
    
    Returns:
        float: rate of change of Recovered.
    
    Examples:
    
        >> dR_dT(I=0.01, recov=0.23)
        0.0023
    """
    return death * I  # Anyone who's Infected can Recover.

def SIR(t, y):
    """
    This function specifies a system of differential equations to be solved,
    and their parameters. We will pass this to the solve_ivp [1]_ function
    from the scipy library.
    
    Args:
        t (float): time step.
        y (list):  parameters, in this case a list containing [S, I, R, transm, recov].
        
    Returns:
        list: Calculated values [S, I, R, transm, recov]
    
    Examples:
        
        >>> SIR(t=0, y=[0.99, 0.01, 0.0, 3.2, 0.23])
        [-0.03168, 0.02938, 0.0023]
    
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """
    S, I, D = y
    return [
        dS_dT(S, I, transm, recov),
        dI_dT(S, I, transm, death, recov),
        dD_dT(I, death),
    ]


solution = solve_ivp(
    fun=SIR,  # input function
    t_span=[0, maxT],  # start at time 0 and continue until we get to maxT
    t_eval=np.arange(0, maxT, 0.1),  # points at which to store the computed solutions
    y0=[Sstart, Istart, Rstart],  # initial conditions
)
solution

def plot_curves(solution, xlim=[0, 10], title=None, add_background=True):
    """Helper function that takes a solution and optionally visualises it
       using official Numberphile brown paper.
    
    Args:
        solution (scipy.integrate._ivp.ivp.OdeResult): Output of solve_ivp() function.
        xlim (list): x-axis limits in format [min, max].
        title (str): Optional graph title.
        add_background (bool): Add Numberphile brown paper background?
    
    Returns:
        matplotlib graph of SIR model curves.
    
    Examples:
    
        >>> solution = solve_ivp(SIR, t_span=[0, maxT], t_eval=np.arange(0, maxT, 0.1),
                                 y0=[Sstart, Istart, Rstart])
        >>> plot_curves(solution, title="The SIR Model of disease spread")
    """
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.title(title, fontsize=15)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Percentage of population", fontsize=12)
    # Create DataFrame
    df = pd.DataFrame(
        solution.y.T,
        columns=["Susceptible", "Infected", "Dead"],
        index=solution.t,
    )
    # Make the plot
    plot = df.plot(color=["blue", "red", "green"], lw=2, ax=ax)
    plot.set_xlim(xlim[0], xlim[1])
    # Add background?
    #if add_background:
        #plot.imshow(
            #background,
            #aspect=plot.get_aspect(),
            #extent=plot.get_xlim() + plot.get_ylim(),
            #zorder=1,
        #)


#plot_curves(solution, title="The SIR Model of disease spread")

def solve_and_plot(
    Istart=0.01,
    Rstart=0,
    transm=3.2,
    recov=0.23,
    maxT=20,
    title=None,
    add_background=False,
):
    """Helper function so we can play around with the parameters using the interact ipywidget.
    
    Args:
        Istart (float): Starting value for Infected (as percent of population).
        Rstart (float): Starting value for Recovered (as percent of population).
        transm (float): transmission rate.
        recov (float): recovery rate.
        maxT (int): maximum time step.
        title (str): Optional graph title.
        add_background (bool): Optionally add Numberphile background.
    
    Returns:
        matplotlib graph of SIR model curves.
    
    Examples:
    
        >>> solve_and_plot(maxT=20, title="Set maxT = 20")
    """


    N = 1
    Sstart = N - Istart

    def SIR(t, y):
        """We need to redefine this inside solve_and_plot() otherwise it
           won't pick up any changes to transm or recov.
        """
        
        if 2<=t<=3:
            Quaren = 0.001 * transm
        else:
            Quaren = transm

        
        S, I, D = y
        return [
            dS_dT(S, I, Quaren, recov),
            dI_dT(S, I, Quaren, death, recov),
            dD_dT(I, death),
        ]

    solution = solve_ivp(
        fun=SIR,
        t_span=[0, maxT],
        t_eval=np.arange(0, maxT, 0.1),
        y0=[Sstart, Istart, Rstart],
    )
    plot_curves(solution, xlim=[0, maxT], title=title, add_background=add_background)


# Let's set maxT to 20 to see how things pan out
solve_and_plot(maxT=20, title="Set maxT = 20")


