import numpy as np
import ctypes

ftype = np.float32
ctype = np.complex64
chalf = (ctypes.c_float * 2)(0.5, 0)
zhalf = (ctypes.c_double * 2)(0.5, 0)

PI      = ftype(np.pi)
C_LIGHT = ftype(137.035999084)  # speed of light
AU2EV   = ftype(27.211386245988)  # Hartree to eV
AU2A    = ftype(0.52917721092)     # Bohr to Angstrom

