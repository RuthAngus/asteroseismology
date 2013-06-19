PyBLS
=====

Python wrapper for a multithreaded F2003 version of the EEBLS routine by Kovacs et al. (Kovacs, Zucker & Mazeh 2002, A&A, Vol. 391, 369). The original F77 routine can be found from http://www.konkoly.hu/staff/kovacs/index.html.


Install
-------

	setup.py config_fc --fcompiler=gnu95 --opt="-Ofast -ffast-math" --f90flags="-cpp -fopenmp -march=native" install --user
