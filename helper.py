import numpy as np

def readData(fp,dtype):
    """Read records from Fortran binary file and convert to
    np.array of given dtype."""
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


def symmetrize_axial_tensor(M: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Symmetrize a second-rank axial tensor M using symmetry operations R.

    The symmetrized tensor is computed as:
        M_sym = (1 / nr) * Σ_R [det(R) * (R @ M @ R^-1)]

    Args:
        #M (np.ndarray): The axial tensor(s), shape (..., 3, 3)
        # we have not encountered other shapes.
        M (np.ndarray): The axial tensor(s), shape (ne, 3, 3)
        R (np.ndarray): Rotation matrices, shape (nr, 3, 3)

    Returns:
        #np.ndarray: Symmetrized tensor(s), shape (..., 3, 3)
        np.ndarray: Symmetrized tensor(s), shape (ne, 3, 3)
    """
    R = R.astype(M.dtype)
    
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("R must be of shape (nr, 3, 3)")
    if M.shape[-2:] != (3, 3):
        raise ValueError("M must be a 3x3 tensor or batch of 3x3 tensors")

    nr = R.shape[0]
    det_R = np.linalg.det(R)  # Shape: (nr,)

    # Scale each R by its determinant for axial tensor symmetrization
    R_scaled = R * det_R[:, np.newaxis, np.newaxis]

    # Perform R * M * R.T with broadcasting and summation over all symmetry operations
    #sym_M = np.einsum('nij,...jk,nlk->...il', R_scaled, M, R, optimize=True)
    # two-step is more efficient
    #R_scaled_M = np.einsum('nij,...jk->...nik', R_scaled, M, optimize=True)
    #sym_M = np.einsum('...nik,nlk->...il', R_scaled_M, R, optimize=True)

    # matmul and tensordot is always preferred over einsum

    R_scaled_M = np.tensordot(R_scaled, M, axes=(2, 1))
    sym_M = np.tensordot(R_scaled_M, R, axes=([0,3],[0,2])).swapaxes(0,1)

    # Average over the symmetry operations
    return sym_M / nr


def finite_difference(f, d, axis, order=2):
    """
    Compute the first derivatives along axis using high-order finite differences
    with periodic boundary conditions.
    
    Args:
        f (np.ndarray): Scalar field values.
        d (float): Grid spacings along axis.
        order (int): Finite difference order (2, 4, 6, or 8).
        
    Returns:
        np.ndarray: df/d, with the same shape as f.
    """
    if order == 2:
        shifts = [-1, 1]
        coeffs = np.array([1, -1]) / 2
    elif order == 4:
        shifts = [-2, -1, 1, 2]
        coeffs = np.array([-1, 8, -8, 1]) / 12
    elif order == 6:
        shifts = [-3, -2, -1, 1, 2, 3]
        coeffs = np.array([1, -9, 45, -45, 9, -1]) / 60
    elif order == 8:
        shifts = [-4, -3, -2, -1, 1, 2, 3, 4]
        coeffs = np.array([-3, 32, -168, 672, -672, 168, -32, 3]) / 840
    else:
        raise ValueError("Only orders 2, 4, 6, or 8 are supported.")
    df = np.zeros_like(f)
    for shift, c in zip(shifts, coeffs):
        df += c * np.roll(f, shift=shift, axis=axis)
    return df / d





    
