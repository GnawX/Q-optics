# Q-optics

**Q-optics** computes optical response functions associated with **spatial dispersion** (finite wavevector **q**), by evaluating the **first-order expansion in q** of the dielectric tensor $\boldsymbol{\epsilon}_{ij}(\omega,\mathbf{q})$. Currently, the natural optical activity (NOA) (**optical rotation** and **circular dichroism**) tensor is implemented. In this framework, NOA is obtained from the **antisymmetric** part of the dielectric tensor. The **symmetric** part contains additional spatial-dispersion phenomena such as **gyrotropic birefringence** and **nonreciprocal directional dichroism**, which can be enabled with minor extensions to the current implementation.

## Notes

For the current implementation, the only inputs to calculate NOA is single particle energy $E_{n\mathbf{k}}$ and velocity matrix elements $\mathbf{V}_{nm\mathbf{k}}$, as well as exciton envelope $A_{cv\mathbf{k}}^\lambda$ and energy $\Omega_\lambda$ if excitonic effect are to be included.

## Dependencies

- `numpy`
- `spglib` (symmetrization)
- `phonopy` (reading `POSCAR`)  
  followings are needed for MPI parallelization of BSE calculations.
- `MPI4py`
- `Pyscalapack`

