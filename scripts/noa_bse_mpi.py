import os
import time
import numpy as np
import gc
from mpi4py import MPI
from bloch import Electrons
from lattice import Lattice
from bse import BSE
#from bse_mpi import calc_electric_multipole_bse
from bse_mpi import calc_electric_multipole_bse_scal

# Set environment variables for OpenMP/MKL before importing NumPy
# Set in batch script
#os.environ["OMP_NUM_THREADS"] = 
#os.environ["MKL_NUM_THREADS"] = 
os.environ["PYTHONUNBUFFERED"] = "1"

# setting parameters
emin, emax, ne = 0, 10, 1001      # energy grid
eta = 0.01                        # smearing
delta, kc, kv = 0.51, 1.0, 1.0    # scissors correction

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    t0 = time.time()

    # Data to broadcast
    broadcast_data = None

    # Root process: Initialize data
    if rank == 0:
        t_start = time.time()
        electron = Electrons('EIGENVAL', 'std')
        electron.apply_scissors_correction(delta, kc, kv)

        lat = Lattice('POSCAR')

        rmn = electron.read_waveder('WAVEDER')
        eig_der = electron.read_eigder('EIGDER')
        vmn = electron.calc_velocity_matrix_elements(rmn, eig_der)

        exciton = BSE('EXCEIG')
        exciton.convert_symop_cryst_to_cart(lat.acell)
        vcv, mcv, qcv = exciton.calc_electric_multipole_ipa(rmn, vmn)
        del rmn, vmn
        gc.collect()


        #broadcast_data = {
        #    'table': exciton.table2,'vcv': vcv, 'mcv': mcv, 'qcv': qcv
        #}
        broadcast_data = {
            'vcv': vcv, 'mcv': mcv, 'qcv': qcv
        }
        print(f"{'Initialize:':30s} {time.time() - t_start:8.2f} seconds", flush=True)

    # Broadcast all data in one go
    comm.Barrier()  # Ensure root is ready
    t_bcast = time.time()
    broadcast_data = comm.bcast(broadcast_data, root=0)
    if rank == 0:
        print(f"{'Brodcast:':30s} {time.time() - t_bcast:8.2f} seconds", flush=True)

    # Unpack broadcasted data
    vcv, mcv, qcv = broadcast_data['vcv'], broadcast_data['mcv'], broadcast_data['qcv']
    # table = broadcast_data['table']

    # Parallel computation
    t_calc = time.time()
    #V, M, Q = calc_electric_multipole_bse(table, vcv, mcv, qcv, comm)
    V, M, Q = calc_electric_multipole_bse_scal(vcv, mcv, qcv, comm)
    if rank == 0:
        print(f"{'V, M, Q:':30s} {time.time() - t_calc:8.2f} seconds", flush=True)

    # Root process: Final computation
    #if rank == 0:
        t_opt = time.time()
        exciton.calc_natural_optical_activity(
            V, M, Q, vol=lat.volume, nspinors=1,
            emin=emin, emax=emax, ne=ne, eta=eta, rotations=lat.rot_cart
        )
        print(f"{'Gamma:':30s} {time.time() - t_opt:8.2f} seconds", flush=True)

    comm.Barrier()  # Synchronize for final timing
    if rank == 0:
        print(f"{'Total time elapsed:':30s} {time.time() - t0:8.2f} seconds", flush=True)

if __name__ == "__main__":
    main()
