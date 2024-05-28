#!/usr/bin/env python

"""
This is the implementation to the final project

TODO:
- performance statistics
"""


import time
import numpy as np
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools

import scipy as sp
from scipy.sparse import identity, eye, spdiags
from scipy.sparse.linalg import gmres, spilu, LinearOperator

# for the plotting
markers = itertools.cycle(['o', 's', '^', 'D', '*', 'x', '+', 'p', 'h', 'v'])

# path for figures
figpath = '../Assignments/FinalProject/plots'

# switch between explicit and implicit implementation
implicit = True

class Problem:
    def __init__(self,dt=1e-4,dx=1e-4,n=None):
        # define timestep
        self.dt = dt
        if n:
            self.n = n
            self.dx = 2/n
        else:
            self.n = int(2/dx)
            self.dx = dx
        # create space
        self.X = np.linspace(0,2,self.n)

    def rhs_implicit(self,un,unm1):
        un_shift = np.roll(un, 1)
        return un-unm1+self.dt/(2*self.dx)*(un-un_shift)*(un+un_shift)

    def jac_implicit(self,u):
        return A_burger(u,self.dt/self.dx)

def A_burger(u,dt_dx):
    n = u.shape[0]
    return spdiags([-dt_dx*u,1+dt_dx*u,-dt_dx*u],[-1,0,n])

def initialize(n=2**11):
    # define space
    prb = Problem((1e-2 if implicit else 1e-4), n=n)
    dx = prb.dx

    # initial condition
    def u0(x):
        return 2+np.sin(np.pi*x)

    un = (u0(prb.X-dx/2)+u0(prb.X+dx/2))/2
    return prb, un

gamma = 1 # V-cycle
nu_1 = 3 # number of presmoothing steps
nu_2 = 3 # number of postsmoothing steps
maxlevels = 7 # maximum number of levels before exact solve

def smoother_jacobi(x,b,u,nu,dt_dx_l):
    # using Jacobi
    n = u.shape[0]
    Al_offdiag = spdiags([-dt_dx_l*u,-dt_dx_l*u],[-1,n])
    Al_diag = 1+dt_dx_l*u
    for i in range(nu): x = (b-Al_offdiag@x)/Al_diag
    return x

def smoother_RK(x,b,u,nu,dt_dx_l):
    c = 1/3 # take into account that |u|<=3
    alphas = [1/3,1]
    dt_star = c/dt_dx_l
    A = A_burger(u,dt_dx_l)
    for i in range(nu):
        yj = x.copy()
        for alpha in alphas:
            yj = x - alpha*dt_star*(A@yj-b)
        x = yj
    return x

def restrict(x):
    return (x[::2]+x[1::2])/2

def prolong(x):
    x_new = np.zeros(2*x.shape[0])
    x_new[::2] = x
    x_new[1::2] = x
    return x_new

def multigridStep(x,b,u,l,dt_dx):
    dt_dx_l = 2**(maxlevels-l)*dt_dx # dt/dx for step l
    if l==0:
        # solve exactly
        return sp.sparse.linalg.spsolve((A_burger(u,dt_dx_l)),b)
    # presmoothing
    x = smoother_RK(x,b,u,nu_1,dt_dx_l)
    # restrict residual
    r = restrict(A_burger(u,dt_dx_l)@x - b)
    v = np.zeros_like(r)
    ulm1 = restrict(u)
    for i in range(gamma): v = multigridStep(v,r,ulm1,l-1,dt_dx)
    # correct fine grid error
    x -= prolong(v)
    # postsmoothing
    x = smoother_RK(x,b,u,nu_2,dt_dx_l)
    return x

default_params = {
    'precond': 'multigrid',
    'eisenwalker': True,
    'name': '',
    'showfig': True,
    'implicit': True
}

def evolve_implicit(prb, un, l, plot_steps=None, ax_gmres_res=None, params=default_params):
    ####################################
    #
    # Newton-GMRES
    #
    ####################################
    dx = prb.dx
    dt = prb.dt
    n = prb.n

    uk = np.copy(un)             # Initial guess
    bk = prb.rhs_implicit(uk, un)     # Initial rhs
    res_k = np.linalg.norm(bk)   # Initial residual
    res_km1 = res_k

    gamma = 0.5 # 0<gamma<=1
    eta_max = 0.5 # eta_max<1
    eta_k = eta_max
    eta_km1 = eta_k

    residuals = [] # list of residuals
    gmres_iters = [] # list of gmres iterations
    k = 0

    # For the Newton iterations
    Newton_tol = 1e-10

    while res_k > Newton_tol:
        # Jacobian Ak at current iterate
        Ak = prb.jac_implicit(uk)
        k = len(residuals)

        # preconditioning with multigrid
        x0 = np.zeros_like(uk)

        precond = params['precond']
        if precond == 'none':
            M = eye(n)
        elif precond == 'multigrid':
            M = LinearOperator((n,n),lambda x: multigridStep(x0,x,uk,maxlevels,dt/dx))
        elif precond == 'jacobi':
            M = LinearOperator((n,n),lambda x: x/Ak.diagonal())

        A = LinearOperator((n,n), lambda x: M*(Ak.dot(x)))

        # update eta_k for Eisenstat and Walker
        etaA_k = gamma*(res_k/res_km1)**2
        gammaeta_km1 = gamma*(eta_km1**2)
        if k == 0:
            etaC_k = eta_max
        else:
            if gammaeta_km1 <= 0.1:
                etaC_k = min(eta_max, etaA_k)
            else:
                etaC_k = min(eta_max, max(etaA_k, gammaeta_km1))
        eta_km1 = eta_k

        if not params['eisenwalker']:
            eta_k = Newton_tol # default condition
        else:
            eta_k = min(eta_max,max(etaC_k,Newton_tol/(2*res_k))) # Eisenstat and Walker

        # Initialize a counter for iterations
        global num_iterations
        num_iterations = 0
        global residuals_gmres
        residuals_gmres = []

        # Define a callback function that increments the iteration counter
        def callback(x):
            global num_iterations
            num_iterations += 1
            global residuals_gmres
            residuals_gmres += [np.linalg.norm(bk+Ak.dot(x))]

        # Solve linear system using GMRES
        du,_ = gmres(A, -bk, callback=callback, tol=eta_k, atol=0, callback_type='x')

        # Update Newton solution and residual
        uk += du
        bk = 0*(uk - un) + prb.rhs_implicit(uk, un)
        res_k = np.linalg.norm(bk)

        # performance statistics
        residuals += [res_k]
        gmres_iters += [num_iterations]

        # plotting performance
        if l == 2:
            marker = next(markers)
            residuals_gmres = np.array(residuals_gmres)
            ax_gmres_res.scatter(range(len(residuals_gmres)),residuals_gmres/residuals_gmres[0],label=f"Newton step {k}", marker=marker)
        
        k+=1
        
    un = np.copy(uk)

    # return performance statistics
    return un, residuals, gmres_iters

def evolve_explicit(prb, un):
    dt = prb.dt
    dx = prb.dx
    un_shift = np.roll(un, 1)
    return un-dt/dx*(un-un_shift)*(un+un_shift)

def compute(params=default_params, prb = None, un = None):

    # initialize solution with initial data
    if prb == None: prb, un = initialize()


    # performance statistics
    nr_gmres  = [] # list of number of gmres iterations needed
    nr_newton  = [] # list of number of newton iterations needed

    plot_steps = [0,2,5,8,10] # list of steps in which to show a plot

    # initialise plots
    fig_gmres_res, ax_gmres_res = plt.subplots()
    fig_gmres, ax_gmres = plt.subplots()
    fig_newton, ax_newton = plt.subplots()

    X = prb.X
    dt = prb.dt

    plot_state(X, un)

    un_list = [un]

    t = 0.
    T = 0.1 if implicit else 2 # end time
    l = 0 # time step

    start = time.time()

    # main solver loop
    while t < T:
        # evolve to next time step
        if params['implicit']:
            un, residuals, gmres_iters = evolve_implicit(prb, un, l, plot_steps, ax_gmres_res, params)
            # performance statistics
            nr_newton += [len(residuals)]
            nr_gmres += [np.sum(np.array(gmres_iters))]

            # plotting performance
            if l in plot_steps:
                marker = next(markers)
                residuals = np.array(residuals)
                ax_newton.scatter(range(nr_newton[-1]),residuals,label=f"Time step {l}", marker=marker)
                ax_gmres.scatter(range(nr_newton[-1]),gmres_iters,label=f"Time step {l}", marker=marker)
        else:
            un = evolve_explicit(prb, un)
        un_list += [un.copy()]

        l += 1
        t += dt
        print(f"Time = {t}, dt = {dt}")

    end = time.time()
    print("time used (Python):", end-start)

    # beautify performance plots
    ax_newton.set_xlabel ('Newton step')
    ax_newton.set_ylabel ('Residual')
    ax_newton.set_yscale('log')
    ax_newton.legend()
    ax_gmres.set_xlabel ('Newton step')
    ax_gmres.set_ylabel ('Number of GMRES iterations')
    ax_gmres.set_ylim(bottom=0)
    ax_gmres.legend()
    ax_gmres_res.set_xlabel ('GMRES step')
    ax_gmres_res.set_ylabel ('Relative residuals $|r_j|/|r_0|$')
    ax_gmres_res.set_yscale('log')
    ax_gmres_res.legend()

    # plot final solution
    Y = np.linspace(0,T,len(un_list))
    Z = np.array(un_list)
    plot_solution(X, Y, Z)

    # create more statistics plots
    fig_nr_newton, ax_nr_newton = plt.subplots()
    fig_nr_gmres, ax_nr_gmres = plt.subplots()

    for ax, Y, text in [[ax_nr_newton, nr_newton, 'Newton'],[ax_nr_gmres, nr_gmres, 'GMRES']]:
        X = range(len(Y))
        ax.scatter(X,Y)
        ax.set_xlabel('Time step')
        ax.set_ylabel(f'Number of {text} iterations')
        ax.set_ylim(bottom=0)
        ax.set_xticks(X)
    
    if params['name']:
        # save statistics plots
        for fig, name in [[fig_gmres_res,'gmres_res'],[fig_gmres, 'gmres_nr'],[fig_newton, 'newton_res'],[fig_nr_newton, 'newton_nr'],[fig_nr_gmres, 'newton_nr']]:
            fig.savefig(f"{figpath}/{name}_{params['name']}.pdf")

    if params['showfig']:
        plt.show()
    else:
        plt.close('all')

    # return statistics
    return np.array(nr_newton), np.array(nr_gmres)


def plot_state(X,Y):
    # Create a plot of the state at fixed time
    fig = plt.figure()
    plt.plot(X,Y)

def plot_solution(X,Y,Z):
    # reduce the resolution
    x_stride = int(len(X)/1000)+1
    y_stride = int(len(Y)/1000)+1
    X = X[::x_stride].copy()
    Y = Y[::y_stride].copy()
    Z = Z[::y_stride,::x_stride].copy()

    # Create the plot
    fig = plt.figure()
    plt.subplot(projection='3d')
    ax = fig.gca()
    Xm, Ym = np.meshgrid(X,Y)
    ax.plot_surface(Xm,Ym,Z,cmap=cm.coolwarm)
    ax.set(
        xlabel='x',
        ylabel='t',
        zlabel='u'
    )


# main part
if __name__ == '__main__':

    if False:
        compute()

    if True:
        for precond in ['none','jacobi','multigrid']:
            for eisenwalker in [True,False]:
                name = precond
                name += '_eisenwalker' if eisenwalker else '_fixed'
                params = {
                    'precond': precond,
                    'eisenwalker': eisenwalker,
                    'name': name,
                    'showfig': True,
                    'implicit': True
                }
                nr_newton, nr_gmres = compute(params)


    if False:
        # check for mesh independent convergence
        nr_newtons = []
        nr_gmress = []
        nr_gmres_newtons = []

        N = range(7,14)
        for n in N:
            prb, un = initialize(2**n)
            params = default_params
            params['showfig'] = False
            nr_newton, nr_gmres = compute(params, prb, un)

            nr_newtons += [np.sum(nr_newton)]
            nr_gmress += [np.sum(nr_gmres)]
            nr_gmres_newtons += [np.mean(nr_gmres/nr_newton)]
        
        for Y, y_label, name in [[nr_newtons, 'Number of Newton iterations', 'nr_newtons'], [nr_gmress, 'Number of GMRES iterations', 'nr_gmres'], [nr_gmres_newtons, 'average GMRES iterations / Newton step', 'gmres_newton']]:
            fig, ax = plt.subplots()
            ax.scatter(N,Y)
            ax.set_xlabel('$log_2$(mesh size)')
            ax.set_ylabel(y_label)
            ax.set_ylim(bottom=0)
            ax.set_xticks(N)

            # save figure
            if True:
                plt.show()
            if True:
                fig.savefig(f'{figpath}/mesh_independent_{name}')



