import os

# Set environment variables for OpenMP/MKL 
num_threads = os.cpu_count() or 16  # Fallback to 16 if cpu_count is None
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

import time
from bloch import Electrons
from lattice import Lattice

# Energy grid and parameters
emin, emax, ne = 0, 5, 1001  # Energy grid
eta = 0.1                    # Smearing
delta, kc, kv = 0.0, 1.0, 1.0  # Scissors correction

def main():
    t0 = time.time()

    # Initialize electrons
    t_start = time.time()
    electron = Electrons('EIGENVAL', 'std')
    # electron.apply_scissors_correction(delta, kc, kv)  # Disabled as delta=0
    print(f"Electron initialization time: {time.time() - t_start:.2f} seconds")

    # Initialize lattice
    t_start = time.time()
    lat = Lattice()
    rotations = lat.rot_cart
    print(f"Lattice initialization time: {time.time() - t_start:.2f} seconds")

    # Read wavefunction derivatives and compute velocity matrix
    t_start = time.time()
    R = electron.read_waveder('WAVEDER')
    eig_der = electron.read_eigder('EIGDER')
    V = electron.calc_velocity_matrix_elements(R, eig_der)
    print(f"Velocity matrix computation time: {time.time() - t_start:.2f} seconds")

    # Compute magnetic dipole and electric quadrupole
    t_start = time.time()
    M, Q = electron.calc_magnetic_dipole_electric_quadrupole(R, V)
    print(f"Magnetic dipole and electric quadrupole computation time: {time.time() - t_start:.2f} seconds")

    # Compute natural optical activity
    t_start = time.time()
    electron.calc_natural_optical_activity(
        V, M, Q, vol=lat.volume,
        emin=emin, emax=emax, ne=ne, eta=eta, rotations=rotations
    )
    print(f"Natural optical activity time: {time.time() - t_start:.2f} seconds")

    print(f"Total time elapsed: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
