# Q-optics

**Q-optics** computes optical response functions associated with **spatial dispersion** (finite wavevector **q**), by evaluating the **first-order expansion in q** of the dielectric tensor $\boldsymbol{\epsilon}_{ij}(\omega,\mathbf{q})$. Currently, the natural optical activity (NOA) (**optical rotation** and **circular dichroism**) tensor is implemented. In this framework, NOA is obtained from the **antisymmetric** part of the dielectric tensor. The **symmetric** part contains additional spatial-dispersion phenomena such as **gyrotropic birefringence** and **nonreciprocal directional dichroism**, which can be enabled with minor extensions to the current implementation.

## Dependencies

- `numpy`
- `spglib` (symmetrization)
- `phonopy` (reading `POSCAR`)  
  followings are needed for MPI parallelization of BSE calculations.
- `MPI4py`
- `Pyscalapack`

