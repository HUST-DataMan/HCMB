"""
This is an an algorithm, Hi-C Matrix Balancing (HCMB),
based on iterative solution of equations, combining with linear
search and projection strategy to normalize the Hi-C original interaction data
with highly sparse

Version 1.0
"""

import gcMapExplorer.lib as gmlib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
#%matplotlib inline
plt.style.use('ggplot')

import numpy.linalg as LA
import scipy.linalg as SLA
import time
from sys import stdout


# F function and its jacobian function
def F(x, A):
    # or x * A.dot(x)  x列向量
    e = np.ones(len(x))
    return np.dot(np.diag(x), A).dot(x) - e


def JF(x, A):
    return np.dot(np.diag(x), A) + np.diag(np.dot(A, x))


# merit function
def f(x, A):
    return LA.norm(F(x, A)) ** 2 / 2.


def gradf(J, F_1):
    #    return np.dot(JF(x,A).T, F(x,A))
    return np.dot(J.T, F_1)


def proj_plus(x):
    return np.maximum(x, 0)


def update_x_next(x, alpha, dk):
    # x current vector; t stepsize ; d  search direction
    return proj_plus(x + alpha * dk)


def gen_stepsize_PG(x_current, A, dk, sigma, beta):
    assert sigma < 0.5, "Armijo rule: sigma less than 0.5"
    assert beta < 1, "Decay factor less than 1"
    gradf_current = np.dot(JF(x_current, A).T, F(x_current, A))
    f_current = f(x_current, A)
    alpha = 1
    # dk = -gradf_current
    x_next = update_x_next(x_current, alpha, dk)

    while True:
        if np.isnan(f(x_next, A)):
            alpha *= beta
        else:
            if f(x_next, A) >= f_current + sigma * np.dot(gradf_current.T, x_next - x_current):
                alpha *= beta
            else:
                break
        if alpha < 1e-16:
            break
        x_next = update_x_next(x_current, alpha, x_next - x_current)
    return alpha


def gen_stepsize_LS(x_current, A, dk, sigma, beta):
    assert sigma < 0.5
    assert beta < 1
    gradf_current = np.dot(JF(x_current, A).T, F(x_current, A))
    f_current = f(x_current, A)
    alpha = 1

    sk = proj_plus(x_current + dk) - x_current

    #    if gradf_current.dot(sk) > -rho*LA.norm(sk)**p: # sk is not a descent direction
    #        return -1

    x_next = x_current + alpha * sk

    while True:
        if np.isnan(f(x_next, A)):
            alpha *= beta
        else:
            if f(x_next, A) >= f_current + sigma * np.dot(gradf_current.T, x_next - x_current):
                alpha *= beta
            else:
                break
        if alpha < 1e-16:
            # 1e-12
            break
        x_next = x_current + alpha * sk
    return alpha

#the main code of HCMB algorithm
#
def HCMB(A, xk=None, kmax=10000, eps=1e-5, tmin=1e-16, verbose=False):
    # (A, xk=None, kmax=500, eps=1e-4, tmin=1e-10, verbose=False):
    """ HCMB implementation
    """

    beta = 0.9;
    sigma = 1e-4;
    gama = 0.99995;
    mu = 1;
    rho = 1e-8;
    p = 2.1
    k = 0
    n = A.shape[0]
    e = np.ones(n) * 1
    if xk is None:
        xk = e
    x_next = np.zeros(n)
    muk = 0.5 * 1e-8 * LA.norm(F(xk, A)) ** 2  # initally muk

    while k < kmax:

        Fxk = F(xk, A)
        Hk = JF(xk, A)

        a = np.dot(Hk.T, Hk) + muk * np.eye(len(xk))
        b = -np.dot(Hk.T, Fxk)
        # a * d = b
        try:
            dk = SLA.solve(a, b, assume_a='pos')
            x_tmp = proj_plus(xk + dk)
        except SLA.LinAlgError:
            print("Error: Singular Matrix")
            return -1
        except SLA.LinAlgWarning:
            print("Warning: ill-conditioned matrix")
            x_next = LA.inv(np.diag(np.dot(A, xk))).dot(np.ones(n))  # restart x_0

        sk = x_tmp - xk
        S25 = (LA.norm(F(x_tmp, A)) <= gama * LA.norm(Fxk))

        if S25:
            x_next = x_tmp
        elif (not S25) and (np.dot(gradf(Hk, Fxk).T, sk) <= -rho * LA.norm(sk) ** p):
            # sk is a descent direction, use armijo line search to reduce f
            t_LS = gen_stepsize_LS(xk, A, dk, sigma, beta)
            print("LS stepsize: %f" % t_LS)
            x_next = xk + t_LS * sk
        else:
            dk = -gradf(Hk, Fxk)
            t_PG = gen_stepsize_PG(xk, A, dk, sigma, beta)
            print("PG stepsize: %f" % t_PG)
            # x = proj_plus(xk - t*gradf(Hk, Fxk))
            x_next = update_x_next(xk, t_PG, dk)

        #        if t_LS <= tmin or t_PG <= tmin
        #            x_next = LA.inv(np.diag(np.dot(A, xk))).dot(np.ones(n))
        #            print("fix-point interver")

        normF = LA.norm(F(x_next, A))
        rmserror = normF

        #        print(np.abs(F(x_next, A) - F(xk, A)).sum())
        #        if np.abs(F(x_next, A) - F(xk, A)).sum() < 1e-4:
        #            reason = "Fxk not changed"

        #            return rmserror, x_next, reason

        xk = x_next
        muk = np.minimum(muk, mu * normF ** 2)
        k = k + 1

        if verbose:
            print("Iterations: {} RMS: {}".format(k, rmserror))

        if rmserror < eps:
            reason = "Fxk converged to min epsilon"
            return rmserror, x_next, reason

        #        if t_LS <= tmin or t_PG <= tmin:
        #            reason = "tk is too small"
        #            return rmserror, x_next, reason

        Fxk = F(x_next, A)
        Hk = JF(x_next, A)
        if LA.norm(gradf(Hk, Fxk)) <= eps:
            reason = "grad fx converage to zeros"
            return rmserror, x_next, reason

    return rmserror, x_next, "Finished kmax iterations"


# Do HCMB normalize
# input raw matrix
# output normlaize matrix, minvalue, maxvalue,t
def get_norm_HCMB(matrix):
    # Discard rows and columns with missing data
    bNonZeros = gmlib.ccmapHelpers.get_nonzeros_index(matrix)
    A = (matrix[bNonZeros, :])[:, bNonZeros]
    mx = A.shape[0]
    for i in range(mx):
        if A[i, i] == 0.0:
            A[i, i] = 1.0

    x0 = 1. / np.sum(A, axis=1)
    # Calculate normalization vector using HCMB
    a1=time.perf_counter()
    rmserror, x_solve, msg = HCMB(A, xk=x0, verbose=True)
    b1 = time.perf_counter()
    t = b1 - a1

    vector = x_solve.reshape(len(x_solve), 1)

    outA = vector.T * (A * vector)
    bNoData = ~bNonZeros
    normMat = np.zeros(matrix.shape)  # all 0 init matrix

    # Store normalized values to output matrix
    dsm_i = 0
    ox = normMat.shape[0]
    idx_fill = np.nonzero(~bNoData)
    for i in range(ox):
        if not bNoData[i]:
            normMat[i, idx_fill] = outA[dsm_i]
            normMat[idx_fill, i] = outA[dsm_i]
            dsm_i += 1

    # Get the maximum and minimum value except zero
    ma = np.ma.masked_equal(outA, 0.0, copy=False)
    n_min = ma.min()
    n_max = ma.max()

    return normMat, n_min, n_max, t

def removeZerosColsRows(matrix):
    #matrix is symmetric
    bZeros = np.all(matrix == 0.0, axis=0)
    bNonZeros = ~bZeros
    A=(matrix[bNonZeros,:])[:,bNonZeros]
    return A, bNonZeros

# Do KnightRuiz normalize
# input raw matrix
# output normalize matrix, minvalue, maxvalue
def NormalizeKnightRuizV2(matrix, tol=1e-12, x0=None, delta=0.1, Delta=3, fl=0):
    ''' Knight-Ruiz algorithm for matrix balancing
    This code is from Rajendra,K. et al. (2017) Genome contact map explorer: a platform for the comparison,
    interactive visualization and analysis of genome contact maps. Nucleic Acids Research, 45(17):e152.
    '''

    bNoData = np.all(matrix == 0.0, axis=0)
    bNonZeros = ~bNoData
    A = (matrix[bNonZeros, :])[:, bNonZeros]  # Selected row-column which are not all zeros
    # perform KR methods
    n = A.shape[0]  # n = size(A,1)
    e = np.ones((n, 1))  # e = ones(n,1)
    res = []
    if x0 is None:
        x0 = e

    g = 0.9  # Parameters used in inner stopping criterion.
    etamax = 0.1  # Parameters used in inner stopping criterion.
    eta = etamax
    stop_tol = tol * 0.5
    x = x0
    rt = tol ** 2  # rt = tol^2
    v = x * A.dot(x)  # v = x.*(A*x)
    rk = 1 - v
    rho_km1 = np.dot(rk.conjugate().T, rk)  # rho_km1 = rk'*rk
    rout = rho_km1
    rold = rout

    # x, x0, e, v, rk, y, Z, w, p, ap :     vector shape(n, 1) : [ [value] [value] [value] [value] ... ... ... [value] ]
    # rho_km1, rout, rold, innertol, alpha :  scalar shape(1 ,1) : [[value]]

    MVP = 0  # Well count matrix vector products.
    i = 0  # Outer iteration count.

    if fl == 1:
        print('it in. it res')

    while rout > rt:  # Outer iteration
        i = i + 1
        k = 0
        y = e
        innertol = max([eta ** 2 * rout, rt])  # innertol = max([eta^2*rout,rt]);

        while rho_km1 > innertol:  # Inner iteration by CG
            k = k + 1
            if k == 1:
                Z = rk / v  # Z = rk./v
                p = Z
                rho_km1 = np.dot(rk.conjugate().T, Z)  # rho_km1 = rk'*Z
            else:
                beta = rho_km1 / rho_km2
                p = Z + (beta * p)

            # Update search direction efficiently.
            w = x * A.dot((x * p)) + (v * p)  # w = x.*(A*(x.*p)) + v.*p
            alpha = rho_km1 / np.dot(p.conjugate().T, w)  # alpha = rho_km1/(p'*w)
            ap = alpha * p  # ap = alpha*p (No dot function as alpha is scalar)

            # Test distance to boundary of cone.
            ynew = y + ap;
            # print(i, np.amin(ynew), delta, np.amin(ynew) <= delta)
            # print(i, np.amax(ynew), Delta, np.amax(ynew) >= Delta)
            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.nonzero(ap < 0)  # ind = find(ap < 0)
                gamma = np.amin((delta - y[ind]) / ap[ind])  # gamma = min((delta - y(ind))./ap(ind))
                y = y + np.dot(gamma, ap)  # y = y + gamma*ap
                break
            if np.amax(ynew) >= Delta:
                ind = np.nonzero(ynew > Delta)  # ind = find(ynew > Delta);
                gamma = np.amin((Delta - y[ind]) / ap[ind])  # gamma = min((Delta-y(ind))./ap(ind));
                y = y + np.dot(gamma, ap)  # y = y + gamma*ap;
                break
            y = ynew
            rk = rk - alpha * w  # rk = rk - alpha*w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = np.dot(rk.conjugate().T, Z)  # rho_km1 = rk'*Z

        x = x * y  # x = x.*y
        v = x * A.dot(x)  # v = x.*(A*x)
        rk = 1 - v
        rho_km1 = np.dot(rk.conjugate().T, rk)  # rho_km1 = rk'*rk
        rout = rho_km1
        MVP = MVP + k + 1

        # Update inner iteration stopping criterion.
        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat

        # print(i, res_norm)

        if g * eta_o ** 2 > 0.1:
            eta = np.amax([eta, g * eta_o ** 2])  # eta = max([eta,g*eta_o^2])

        eta = np.amax(
            [np.amin([eta, etamax]), stop_tol / res_norm]);  # eta = max([min([eta,etamax]),stop_tol/res_norm]);

        if fl == 1:
            print('%3d %6d %.3e %.3e %.3e \n' % (i, k, res_norm, np.amin(y), np.amin(x)))
            res = [res, res_norm]

    # Generation of Doubly stochastic matrix ( diag(X)*A*diag(X) )
    outA = x.T * (A * x)

    normMat = np.zeros(matrix.shape)  # all 0 init matrix

    # Store normalized values to output matrix
    dsm_i = 0
    ox = normMat.shape[0]
    idx_fill = np.nonzero(~bNoData)
    for i in range(ox):
        if not bNoData[i]:
            normMat[i, idx_fill] = outA[dsm_i]
            normMat[idx_fill, i] = outA[dsm_i]
            dsm_i += 1

    # Get the maximum and minimum value except zero
    ma = np.ma.masked_equal(outA, 0.0, copy=False)
    n_min = ma.min()
    n_max = ma.max()

    return normMat, n_min, n_max


def plotheatmap(D_mat_real, mapName, method ,title=""):
    fig, ax = plt.subplots()
    m = ax.imshow(D_mat_real, cmap="RdBu_r", norm=colors.LogNorm(),
               origin="bottom",
               extent=(0, D_mat_real.shape[0], 0, D_mat_real.shape[0]))
    cb = fig.colorbar(m)
    #cb.set_label("Contact counts")
    #ax.set_title("HCMB Normalized contact counts", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.savefig('heatmap_{0}_{1}.png'.format(mapName, method), dpi=350)
    plt.show()


def check_symmetric(M, rtol=1e-05, atol=1e-08):
    # M is matrix
    #return True: symmetric
    return np.allclose(M, M.T, rtol=rtol, atol=atol)


def check_result_plot(D_mat_real, mapName, method):
    r_sum = np.sum(D_mat_real, axis=0)
    r_var = np.var(D_mat_real, axis=0)

    c_sum = np.sum(D_mat_real, axis=1)
    c_var = np.var(D_mat_real, axis=1)

    # Plot the values for visual representations
    fig = plt.figure(figsize=(14, 5))  # Figure size
    fig.subplots_adjust(hspace=0.6)  # Space between plots

    fig.suptitle('{0} Normalized Maps of {1}'.format(method, mapName), fontsize=18)

    ax1 = fig.add_subplot(2, 2, 1)  # Axes first plot
    ax1.set_title('Sum along row')  # Title first plot
    ax1.set_xlabel('Position Index')  # X-label

    ax2 = fig.add_subplot(2, 2, 2)  # Axes second plot
    ax2.set_title('Sum along column')
    ax2.set_xlabel('Position Index')

    ax3 = fig.add_subplot(2, 2, 3)  # Axes third plot
    ax3.set_title('Variance along row')
    ax3.set_xlabel('Position Index')

    ax4 = fig.add_subplot(2, 2, 4)  # Axes fourth plot
    ax4.set_title('variance along column')
    ax4.set_xlabel('Position Index')

    ax1.plot(r_sum, marker='o', lw=0, ms=2)  # Plot in first axes
    ax2.plot(c_sum, marker='o', lw=0, ms=2)  # Plot in second axes
    ax3.plot(r_var, marker='o', lw=0, ms=2)  # Plot in third axes
    ax4.plot(c_var, marker='o', lw=0, ms=2)  # Plot in fourth axes

    ax1.get_yaxis().get_major_formatter().set_useOffset(False)  # Prevent ticks auto-formatting
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)

    plt.savefig('checkSum_{0}_{1}.png'.format(mapName, method), dpi=350)
    plt.show()

# compare
def plot_comparison_for_chromosome(matrix1, matrix2, mapName, titleX, titleY):
    import matplotlib as mpl

    fig = plt.figure(figsize=(13, 13))  # Figure size
    fig.subplots_adjust(hspace=0.3, wspace=0.25)  # Space between sub-plots
    mpl.rcParams.update({'font.size': 12})  # Font-size

    # Mask any bins that have zero value, since both map are calculated from same raw map,
    # same bins will have missing data
    matlabNormCCmapMasked = np.ma.masked_equal(matrix1, 0.0, copy=False)
    gcMapExpNormCCmapMasked = np.ma.masked_equal(matrix2, 0.0, copy=False)

    ax = fig.add_subplot(2, 2, 1)  # Axes plot
    ax.set_yscale("log", nonposy='clip')  # Set Y-axis to log scale
    ax.set_xscale("log", nonposx='clip')  # Set X-axis to log scale
    ax.set_title('HCMB vs KR normalized maps on {0}'.format(mapName))  # Title plot

    # Plot for Pearson correlation
    # Plot only those bins that do not have missing data (> zero)
    # 散点图
    ax.scatter(matlabNormCCmapMasked.compressed(), gcMapExpNormCCmapMasked.compressed(), marker='o', s=6)
    #    ax.set_xlabel('HCMB Normalized values')
    #    ax.set_ylabel('KR Normalized values')
    ax.set_xlabel(titleX)
    ax.set_ylabel(titleY)
    plt.savefig('compare_{}.png'.format(mapName), dpi=350)
    plt.show()


