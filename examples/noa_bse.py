import os

# Set environment variables for OpenMP/MKL 
num_threads = os.cpu_count() or 16  # Fallback to 16 if cpu_count is None
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

import time
import numpy as np
from bloch import Electrons
from lattice import Lattice
from bse import BSE

# Energy grid and parameters
emin, emax, ne = 0, 10, 1001   # energy grid
eta = 0.01                     # smearing
delta, kc, kv = 0.51, 1.0, 1.0 # scissors correction

def main():
    t0 = time.time()

    # Initialize electrons and apply scissors correction
    t_start = time.time()
    electron = Electrons('EIGENVAL', 'std')
    electron.apply_scissors_correction(delta, kc, kv)
    print(f"Electron initialization time: {time.time() - t_start:.2f} seconds")

    # Initialize lattice
    t_start = time.time()
    lat = Lattice('POSCAR')
    rotations = lat.rot_cart
    print(f"Lattice initialization time: {time.time() - t_start:.2f} seconds")

    # Read wavefunction derivatives and compute velocity matrix
    t_start = time.time()
    rmn = electron.read_waveder('WAVEDER')
    eig_der = electron.read_eigder('EIGDER')
    vmn = electron.calc_velocity_matrix_elements(rmn, eig_der)
    print(f"Velocity matrix computation time: {time.time() - t_start:.2f} seconds")

    # Initialize BSE and compute amplitudes
    t_start = time.time()
    exciton = BSE('EXCEIG')
    exciton.convert_symop_cryst_to_cart(lat.acell)
    acv = exciton.read_bse_amplitude_multiple('EXCWAV')
    print(f"BSE initialization and amplitude read time: {time.time() - t_start:.2f} seconds")

    # Compute matrix elements
    t_start = time.time()
    V = exciton.calc_velocity_matrix_element(vmn, acv)
    M, Q = exciton.calc_magnetic_dipole_electric_quadrupole(rmn, vmn, acv)
    print(f"Matrix elements computation time: {time.time() - t_start:.2f} seconds")

    # Compute natural optical activity
    t_start = time.time()
    exciton.calc_natural_optical_activity(
        V, M, Q, vol=lat.volume, nspinors=1,
        emin=emin, emax=emax, ne=ne, eta=eta, rotations=rotations
    )
    print(f"Natural optical activity time: {time.time() - t_start:.2f} seconds")

    print(f"Total time elapsed: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
