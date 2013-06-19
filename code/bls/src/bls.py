from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

from time import time
from numpy.ma import masked_array

from blsf import bls

class BLS(object):
    def __init__(self, time, flux, error, **kwargs):
        self.time = time
        self.flux = flux
        self.error = error

        self.fmin = kwargs.get('fmin', 0.1)
        self.nf   = kwargs.get('nf',  1000)
        self.df   = kwargs.get('df',  1e-3)
        self.nbin = kwargs.get('nbin', 1000)
        self.qmin = kwargs.get('qmin', 0.01)
        self.qmax = kwargs.get('qmax', 0.10)


    def __call__(self):
        self.result = BLSResult(*bls.eebls(self.time, self.flux, self.error, self.nf, self.fmin, self.df,
                                          self.nbin, self.qmin, self.qmax))
        return self.result

    
    def get_farray(self):
        return self.fmin + self.df*np.arange(self.nf)


class BLSResult(object):
    def __init__(self, p, bper, bpow, depth, qtran, in1, in2):
        self.p = masked_array(p, p<1e-9)
        self.bper = bper
        self.bpow = bpow
        self.bsde = (bpow - self.p.mean())/(self.p.std()) or 0
        self.depth = depth
        self.qtran = qtran
        self.in1 = in1
        self.in2 = in2
        
    def __str__(self):
        return 'Power {pw:6.4f}   Period {pr:6.3f}   Freq {fr:6.3f}   Depth {df:6.3f}   qtran {qt:5.3f}'.format(pr=self.bper, fr=1/self.bper, pw=self.bpow, df=self.depth, qt=self.qtran)

    def get_sde(self): return (self.p - self.p.mean()) / self.p.std()
