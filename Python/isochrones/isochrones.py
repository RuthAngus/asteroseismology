import numpy as np
import matplotlib.pyplot as pl
import glob

def turnoff_properties(isos):
    # N, Age (Gyr), log Teff, log L/Lsun, Ycore, Mcore
    M = np.zeros(len(isos))
    T = np.zeros(len(isos))
    L = np.zeros(len(isos))
    A = np.zeros(len(isos))

    # find the turnoff temp for each mass
    # find the turnoff lum and age of star for each mass.
    for i, iso in enumerate(isos):
        data = np.genfromtxt(iso, skip_header=1).T
        age = data[1]
        teff = data[2]
        lum = data[3]
        T[i] = max(teff)
        L[i] = lum[teff == max(teff)]
        A[i] = float(age[teff==max(teff)])
        M[i] = .1*float(iso[45:47]) # read from file name

    # For a given mass, what logg and teff does a star have when at MSTO?
    g_0 = 27542.29 # solar properties
    T_0 = 5778 # solar properties
    T = 10**T
    L = 10**L
    logg = np.log10(g_0 * (1./L)*(M)*(T/T_0)**4)

    return M, T, A, L, logg

if __name__ == "__main__":
    isos = np.array(glob.glob("/Users/angusr/angusr/isochrones/a0o2/x53z08/*.track2"))
    M, T, A, L, logg = turnoff_properties(isos)

    pl.clf()
    pl.plot(T, logg, 'k-')
    pl.xlabel('Teff')
    pl.ylabel('logg')
    pl.xlim(4500, 8000)
    pl.xlim(pl.gca().get_xlim()[::-1])
    pl.savefig('teff_vs_logg')

    pl.clf()
    pl.plot(M, A)
    pl.xlabel('mass (Msun)')
    pl.ylabel('turnoff age (Gyr)')
    pl.savefig('turnoff')
