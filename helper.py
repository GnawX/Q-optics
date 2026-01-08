import torch
import numpy as np

def readData(fp,dtype):
    """Read records from Fortran binary file and convert to
    torch.Tensor of given dtype."""
    data = b""
    while True:
        prefix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        data += fp.read(abs(prefix))
        suffix = np.fromfile(fp, dtype=np.int32, count=1)[0]
        if abs(prefix) - abs(suffix):
            raise RuntimeError(
                  "Read wrong amount of bytes.\n"
                  "Expected: %d, read: %d, suffix: %d." % (prefix, len(data), suffix)
            )
        if prefix > 0:
            break
    return np.frombuffer(data, dtype=dtype)


def symmetrize_axial_tensor(M: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize a second-rank axial tensor M using symmetry operations R.

    The symmetrized tensor is computed as:
        M_sym = (1 / nr) * Σ_R [det(R) * (R @ M @ R^-1)]

    Args:
        M (torch.Tensor): The axial tensor(s), shape (..., 3, 3)
        R (torch.Tensor): Rotation matrices, shape (nr, 3, 3)

    Returns:
        torch.Tensor: Symmetrized tensor(s), shape (..., 3, 3)
    """

    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("R must be of shape (nr, 3, 3)")
    if M.shape[-2:] != (3, 3):
        raise ValueError("M must be a 3x3 tensor or batch of 3x3 tensors")
    
    R = R.to(dtype=M.dtype)

    nr = R.shape[0]
    det_R = torch.linalg.det(R)  # Shape: (nr,)

    # Scale each R by its determinant for axial tensor symmetrization
    R_scaled = R * det_R[:, None, None]

    # Perform R * M * R.T with broadcasting and summation over all symmetry operations
    #sym_M = torch.einsum('nij,...jk,nlk->...il', R_scaled, M, R)
    # two-step is more efficient
    R_scaled_M = torch.einsum('nij,...jk->...nik', R_scaled, M)
    sym_M = torch.einsum('...nik,nlk->...il', R_scaled_M, R)

    # Average over the symmetry operations
    return sym_M / nr


    
