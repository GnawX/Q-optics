# Q-optics

**Q-optics** computes optical response functions associated with **spatial dispersion** (finite wavevector **q**), by evaluating the **first-order expansion in q** of the dielectric tensor **$\epsilon_{ij}(\omega,\mathbf{q})$**.

## Implemented features

- **Natural optical activity** (NOA)
  - **Optical rotation**
  - **Circular dichroism**

In this framework, NOA is obtained from the **antisymmetric** part of the dielectric tensor. The **symmetric** part contains additional spatial-dispersion phenomena such as **gyrotropic birefringence** and **nonreciprocal directional dichroism**, which can be enabled with minor extensions to the current implementation.

## Dependencies

- `numpy`
- `spglib` (symmetrization)
- `phonopy` (reading `POSCAR`)

