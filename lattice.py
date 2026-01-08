import numpy as np
import spglib
from consts import ftype
#from ase.io import read as ase_read
from phonopy.interface.calculator import read_crystal_structure


class Lattice:
    """Crystal structure representation from a VASP POSCAR file.

    Attributes:
        filename (str): Path to the POSCAR file.
        atoms: ASE Atoms object containing atomic structure.
        acell (NDArray[np.float64]): Lattice vectors (3x3 matrix, rows as vectors).
        volume (float): Volume of the unit cell.
        spacegroup (str): Space group symbol and number.
        pointgroup (str): Point group symbol.
        translations (NDArray[np.float64]): Translational symmetry operations.
        rot_cryst (NDArray[np.int32]): Rotational symmetry operations in crystal basis.
        rot_cart (NDArray[np.float64]): Rotational symmetry operations in Cartesian basis.
    """

    def __init__(
        self,
        filename: str = "POSCAR",
    ) -> None:
        """Initialize the Lattice object from a POSCAR file.

        Args:
            filename: Path to the POSCAR file (default: 'POSCAR').

        Raises:
            FileNotFoundError: If the POSCAR file cannot be found.
            ValueError: If symmetry analysis fails or data is malformed.
        """

        # Load ASE atoms object with error handling
        try:
            #self.atoms = ase_read(filename)
            phonopyatoms, _ = read_crystal_structure(filename)
            self.atoms = (phonopyatoms.cell, phonopyatoms.scaled_positions, phonopyatoms.numbers)
        except FileNotFoundError:
            raise FileNotFoundError(f"POSCAR file '{filename}' not found.")

        # Lattice vectors (A = [a, b, c]^T, row vector notation)
        self.acell = self.atoms[0].astype(ftype)
        self.volume = ftype(phonopyatoms.volume)

        # Symmetry analysis using spglib
        symmetry = spglib.get_symmetry(self.atoms)
        if symmetry is None:
            raise ValueError("Failed to determine symmetry from POSCAR.")

        self.spacegroup: str = spglib.get_spacegroup(self.atoms)
        self.rot_cryst = symmetry["rotations"]
        self.translations = symmetry["translations"]
        self.pointgroup: str = spglib.get_pointgroup(self.rot_cryst)[0]

        # Convert rotations to Cartesian coordinates: R_cart = A R_cryst A^-1
        acell_t = self.acell.T
        acell_t_inv = np.linalg.inv(acell_t)
        rot_cryst = self.rot_cryst.astype(ftype)
        self.rot_cart = acell_t @ rot_cryst @ acell_t_inv

    def get_rotations_cryst_to_cart(
        self,
        rotations,
    ) -> np.ndarray:
        """Convert rotation matrices from crystal to Cartesian coordinates.

        The transformation is given by R_cart = A @ R_cryst @ A^-1, where A is the
        lattice vector matrix (columns as basis vectors).

        Args:
            rotations: Rotation matrices in crystal coordinates, shape (n_rot, 3, 3).

        Returns:
            Rotation matrices in Cartesian coordinates, shape (n_rot, 3, 3).

        Raises:
            ValueError: If rotations array does not have shape (n, 3, 3).
        """
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(
                f"Expected rotations shape (n, 3, 3), got {rotations.shape}"
            )

        acell_t = self.acell.T
        acell_t_inv = np.linalg.inv(acell_t)
        rotations = rotations.astype(ftype)
        return acell_t @ rotations @ acell_t_inv
