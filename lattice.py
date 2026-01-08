import torch
import spglib
from consts import ftype
from ase.io import read as ase_read

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lattice:
    """Crystal structure representation from a VASP POSCAR file using PyTorch.

    Attributes:
        filename (str): Path to the POSCAR file.
        atoms: ASE Atoms object containing atomic structure.
        acell (torch.Tensor): Lattice vectors (3x3 matrix, rows as vectors).
        volume (float): Volume of the unit cell.
        spacegroup (str): Space group symbol and number.
        pointgroup (str): Point group symbol.
        translations (torch.Tensor): Translational symmetry operations.
        rot_cryst (torch.Tensor): Rotational symmetry operations in crystal basis.
        rot_cart (torch.Tensor): Rotational symmetry operations in Cartesian basis.
    """

    def __init__(self, filename: str = "POSCAR") -> None:
        """Initialize the Lattice object from a POSCAR file."""
        try:
            self.atoms = ase_read(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"POSCAR file '{filename}' not found.")

        # Convert lattice vectors to PyTorch tensor
        self.acell: torch.Tensor = torch.tensor(self.atoms.cell[:], dtype=ftype, device=device)
        self.volume = torch.tensor(self.atoms.get_volume(), dtype=ftype)

        # Symmetry analysis using spglib
        symmetry = spglib.get_symmetry(self.atoms)
        if symmetry is None:
            raise ValueError("Failed to determine symmetry from POSCAR.")

        self.spacegroup: str = spglib.get_spacegroup(self.atoms)

        # Convert symmetry info to PyTorch tensors
        self.rot_cryst: torch.Tensor = torch.tensor(symmetry["rotations"], dtype=torch.int32, device=device)
        self.translations: torch.Tensor = torch.tensor(symmetry["translations"], dtype=ftype, device=device)
        self.pointgroup: str = spglib.get_pointgroup(self.rot_cryst.cpu().numpy())[0]

        # Convert rotations to Cartesian coordinates: R_cart = A R_cryst A^-1
        acell_t = self.acell.T
        acell_t_inv = torch.linalg.inv(acell_t)
        self.rot_cart: torch.Tensor = torch.matmul(acell_t, torch.matmul(self.rot_cryst.to(ftype), acell_t_inv))

    def get_rotations_cryst_to_cart(self, rotations: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices from crystal to Cartesian coordinates."""
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"Expected rotations shape (n, 3, 3), got {rotations.shape}")

        acell_t = self.acell.T
        acell_t_inv = torch.linalg.inv(acell_t)

        return torch.matmul(acell_t, torch.matmul(rotations.to(ftype), acell_t_inv))
