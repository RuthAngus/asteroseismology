import numpy as np
import matplotlib.pyplot as plt
from teff_bv import teff2bv_orig

# from http://irsa.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec
# E(B-V)S&F =  0.86 x E(B-V)SFD
# Schlafly & Finkbeiner 2011 (ApJ 737, 103) (S&F)
# E(B-V) = 0.0359 +/-  0.0012
EBV = .0359

# Schlegel et al. 1998 (ApJ 500, 525) (SFD)
# E(B-V) = 0.0418 +/-  0.0014

# Assuming a visual extinction to reddening
# ratio Av / E(B-V) = 3.1, then:
AvSF =  0.1153 # (mag)
AvSFD =  0.1341 # (mag)

# from http://books.google.de/books?id=aryyroyjkacc&pg=pa137&lpg=pa137&dq=interstellar+extinction+law+binney&source=bl&ots=x9x-f1txix&sig=3_-m2wmcn2aneqicht90lt6clhi&hl=en&sa=x&ei=eeheu4gkia7a4qtop4bi&ved=0cdkq6aewaw#v=onepage&q=interstellar%20extinction%20law%20binney&f=false
# (Binney - Galactic astronomy)
# E(J-V)/E(B-V) = -2.22
EJVEBV = -2.22
# Aj/Av = 0.282
AjAv = .282
Aj = AjAv*AvSF
EJV = EJVEBV*EBV

# E(K-V)/E(B-V) = -2.72
EKVEBV = -2.72
# Ak/Av = 0.112
AkAv = .112
Ak = AkAv*AvSF
EKV = EKVEBV*EBV

#----------------------------------------------------------------------

# Redenning for Coma Ber, from Collier Cameron 2009 (from Nicolet 1981)
# Distance to Coma Ber = 89.9 pm 2.1 parsecs.
EBV = 0.006
EBV_err = 0.013

# from http://stev.oapd.inaf.it/cgi-bin/cmd_2.5 (padova isochrones)
# Recommended by Julianne delcanton
AbAv = 1.29719
AvAv = 1.00600
AjAv = 0.29100
AkAv = 0.11471

# Colour change from extinction:
# EBV = (AbAv - AvAv)*Av
# Colour change due to J-K conversion:
# deltaJK = (AjAv - AkAv)*Av. So
deltaJK = (AjAv - AkAv) * EBV / (AbAv - AvAv)
print deltaJK

# load isochrones
data = np.genfromtxt("baraffe.txt", skip_header=61).T
age = data[4]
teff = data[5]
logg = data[6]
V = data[8]
J = data[11]
K = data[13]

# load Coma Ber
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/ComaBer.txt", skip_header=1).T
j_k = data[2]

# dereddened J-K
j_k = j_k - deltaJK

JK = J-K

# fit a polynomial
a = (.47 < age) * (age < .53) * teff > 3000
JK = J[a] - K[a]
p = np.polyfit(JK, teff[a], 3)
xs = np.linspace(min(JK), max(JK), 1000)
teff_fit = np.polyval(p, xs)

# plot
plt.clf()
plt.plot((JK), teff[a], 'k.')
plt.plot(xs, teff_fit, 'b-')

# find teffs
teff = np.polyval(p, j_k)
print teff

plt.plot(j_k, teff, 'ro')
plt.savefig("JK2teff")

feh = np.zeros_like(teff)
bv = teff2bv_orig(teff, logg, feh)
