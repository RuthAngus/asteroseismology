# Get code working with real data (or at least real x and y) - Teffs will come later
# At the point where it works without noise
# Works with totls as long as the errorbars are small
# Try making errorbars large again but constraining the priors
# Try adding noise without totls, because if that works there may not be a problem!
# Next - get it working with real data!

import numpy as np
import matplotlib.pyplot as pl
import emcee
import scipy.optimize as op
import triangle
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import *
import time

def Gaussian(x, mu, sigma):
    gauss = 1./(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    return gauss/max(gauss)

def totls(x, y, z, xerr, yerr, zerr):
    N = 100
    M = len(x)
    tot_x = np.empty(N*M)
    tot_y = np.empty(N*M)
    tot_z = np.empty(N*M)
    tot_yerr = np.empty(N*M)
    tot_xerr = np.empty(N*M)
    tot_zerr = np.empty(N*M)
    for i in range(M):
        tot_x[i*N:(i+1)*N] = np.random.normal(x[i], xerr[i]/1., N)
        tot_y[i*N:(i+1)*N] = np.random.normal(y[i], yerr[i]/1., N)
        tot_z[i*N:(i+1)*N] = z[i]
        tot_xerr[i*N:(i+1)*N] = xerr[i]
        tot_yerr[i*N:(i+1)*N] = yerr[i]
        tot_zerr[i*N:(i+1)*N] = zerr[i]
    
    return tot_x, tot_y, tot_z, tot_xerr, tot_yerr, tot_zerr

def model(theta, x, bv):
    n, a, b, c = theta
    return n * x + np.log10(a) + b*np.log10(bv - c)

def lnlike(theta, x, y, yerr, bv):
    n, a, b, c = theta
    scaled_residuals = 1.0/(yerr**2) * (y-model(theta, x, bv))**2
    l = np.isfinite(scaled_residuals)
    N = l.sum()
    logL = - 0.5 * float(N) * np.log(2 * np.pi) \
      - np.log(yerr[l]).sum() \
      - 0.5 * scaled_residuals[l].sum()
    return logL

# Flat priors
def lnprior(theta): 
    n, a, b, c = theta
    # if 0.3 < n < 0.7 and 0.5 < a < 0.9 and 0.4 < b < 0.8 and 0.2 < c < 0.5: # Most used priors
    if 0.51 < n < 0.52 and 0.5 < a < 0.9 and 0.4 < b < 0.8 and 0.39 < c < 0.4: # constraining priors
    # if 0.1 < n < 0.8 and 0.3 < a < 1. and 0.2 < b < 0.9 and 0. < c < 0.6: # less constraining priors
    # if 0. < n < 1. and 0. < a < 1. and 0. < b < 1. and 0. < c < 1.: # even less constraining priors
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr, bv):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, bv)

def log_errorbar(y, errp, errm):
    plus = y + errp
    minus = y - errm
    log_err = np.log10(plus/minus) / 2.
    l = minus < 0 # Make sure ages don't go below zero!
    log_err[l] = np.log10(plus[l]/y[l])
    return log_err
    
init = [0.5189,  0.7725, 0.601, 0.4]

# Load data
data = np.genfromtxt('/Users/angusr/Python/Gyro/data/matched_data.txt').T
period = data[1]
p_err = data[2]
age = data[3]*1000 # Convert to Myr
a_err = data[4]*1000
a_errm = data[5]*1000
mass = data[6]
logg = data[9]
a = period > 1. # at least one period measurement is less than one and has a dodgy errorbar
b = logg[a] > 4. # Remove subgiants and Kraft break stars
c = mass[a][b] < 1.3

# Real data
xl = age[a][b][c]
xerrlp = a_err[a][b][c]
xerrlm = a_errm[a][b][c]
# will need to read bv as well from somewhere or calculate it from Teff
yl = period[a][b][c]
yerrlp = p_err[a][b][c]
yerrlm = p_err[a][b][c]

# Fake colours
# xl = np.random.uniform(1000,9000,len(xl)) # Fake x data
bv = np.random.uniform(0.4,1.2,len(xl))
bverr = np.ones_like(bv) * 0.05
l = bv < 0.4
# Resample those points that are less than 0.4
while l.sum() > 0:
    bv[l] = np.random.uniform(0.4,1.2,l.sum())
    bverr[l] = np.ones_like(bv[l]) * 0.005
    l = bv < 0.4
# y = model(init, np.log10(xl), bv) #+ np.random.randn(len(age)) # Fake y data
# yl = 10.0**y
# yerrl = np.ones_like(yl) * 0.1 * yl

# Take logs
x = np.log10(xl)
y = np.log10(yl)
# Calculate logarithmic errorbars
xerr = log_errorbar(xl, xerrlp, xerrlm)
yerr = log_errorbar(yl, yerrlp, yerrlm)

# # Make up errors
# xerr = np.ones_like(xerr) * 0.01
# yerr = xerr

# Resample from gaussian (the totls bit)
x, bv, y, xerr, bverr, yerr = totls(x, bv, y, xerr, bverr, yerr) 

# Run emcee
start = time.time()
ndim, nwalkers, nsteps = 4, 100, 1000
pos = [init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, bv))
sampler.run_mcmc(pos, nsteps)
print 'time = ', (time.time() - start), 'seconds'

# Plot chains
pl.clf()
labels = ['n', 'a', 'b', 'c']
for j in range(len(init)):
    pl.subplot(len(init),1,j)
    [pl.plot(range(nsteps), sampler.chain[i, :, j], 'k-', alpha = 0.2) for i in range(nwalkers)]
    pl.plot(range(nsteps), np.ones(nsteps)*init[j], 'r-')
    pl.ylabel('%s' %labels[j])
pl.savefig('chains')
    
# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print 'initial values', init
mcmc_per = np.array(mcmc_result)[:,2]
mcmc_result = np.array(mcmc_result)[:,0]
print 'mcmc result', mcmc_result

# Triangle plot
pl.clf()
fig = triangle.corner(samples, labels=["$n$", "$a$", "$b$", "$c$"],
                      truths=init)
fig.savefig("triangle.png")

# Calculate maximum-likelihood values
nll = lambda *args: -lnlike(*args)
ml_result = op.fmin(nll, init, args=(x, y, yerr, bv))
print 'ml_result', ml_result

# Sort data
ml_tuples = zip(x, model(ml_result, x, np.ones_like(bv)*bv[0]))
x_ml, y_ml = zip(*(sorted(ml_tuples, key = lambda xy: xy[0])))
init_tuples = zip(x, model(init, x, np.ones_like(bv)*bv[0]))
x_init, y_init = zip(*(sorted(init_tuples, key = lambda xy: xy[0])))
mcmc_tuples = zip(x, model(mcmc_result, x, np.ones_like(bv)*bv[0]))
x_mcmc, y_mcmc = zip(*(sorted(mcmc_tuples, key = lambda xy: xy[0])))

# Plot 2D data. with init, ml and mcmc, x = age, y = period, z = B-V
pl.clf()
xp = np.arange(min(x[np.isfinite(x)]), max(x[np.isfinite(x)]), 0.01)
pl.errorbar(x, y, xerr = xerr, yerr = yerr, fmt = 'k.')
pl.plot(x_ml, y_ml, 'r-', alpha = 0.5)
pl.plot(x_init, y_init, 'k-', alpha = 0.5)
pl.plot(x_mcmc, y_mcmc, 'b-', alpha = 0.5)
pl.ylabel('P')
pl.xlabel('t')
pl.savefig('results')

pl.clf()
fig = pl.figure()
ax = fig.gca(projection='3d')               # to work in 3d
ax.scatter(x, y, bv);                        # plot a 3d scatter plot
ax.set_xlabel('Age')
ax.set_ylabel('Period')
ax.set_zlabel('B-V')
pl.savefig('3dplot')

# Plot parameters
pl.clf()
pl.step(range(0,5), [0, init[0], init[1], init[2], init[3]], color = 'g')
pl.step(range(0,5), [0, ml_result[0], ml_result[1], ml_result[2], ml_result[3]], color = 'r')
pl.step(range(0,5), [0, mcmc_result[0], mcmc_result[1], mcmc_result[2], mcmc_result[3]], color = 'b')
pl.step(range(0,5), [0, mcmc_result[0]+mcmc_per[0], mcmc_result[1]+mcmc_per[1]\
                     , mcmc_result[2]+mcmc_per[2], mcmc_result[3]+mcmc_per[3]], color = 'c')
pl.step(range(0,5), [0, mcmc_result[0]-mcmc_per[0], mcmc_result[1]-mcmc_per[1]\
                     , mcmc_result[2]-mcmc_per[2], mcmc_result[3]-mcmc_per[3]], color = 'c')
pl.savefig('parameters')
