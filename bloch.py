import numpy as np
import torch
from helper import readData, symmetrize_axial_tensor
from consts import ftype, ctype, device, PI, C_LIGHT, AU2A, AU2EV


class Electrons:
    """
    Represents the electronic structure parsed from a VASP EIGENVAL file.

    Attributes:
        nspins (int): Spin polarization tag (1 or 2).
        nelecs (int): Number of electrons.
        nkpts (int): Number of k-points.
        nbands (int): Number of bands.
        kvecs (torch.Tensor): Array of k-point vectors (nkpts, 3).
        weights (torch.Tensor): Weights of each k-point in the Brillouin zone (nkpts,).
        eig (torch.Tensor): Eigenvalues array (nspins, nkpts, nbands).
        occ (torch.Tensor): Occupations array (nspins, nkpts, nbands).
        noccs (torch.Tensor): Number of occupied bands for each spin channel.
        lgamma (bool): Whether gamma-only k-point sampling is used.
        nspinors (int): Number of spinors (1 for collinear spin, 2 for non-collinear).
    """

    def __init__(self, filename: str = 'EIGENVAL', flavor: str = 'std') -> None:
        """
        Initializes the Electrons object by reading a VASP EIGENVAL file.

        Args:
            filename (str): Path to the EIGENVAL file.
            flavor (str): Tag to indicate the format ('std', 'gam', or 'ncl').
        """
        self.lgamma = (flavor == 'gam')
        self.nspinors = 2 if flavor == 'ncl' else 1

        with open(filename, "r") as f:
            # Read spin polarization from header
            self.nspins = int(f.readline().split()[-1])

            # Skip 4 header lines
            for _ in range(4):
                f.readline()

            # Read main parameters
            self.nelecs, self.nkpts, self.nbands = map(int, f.readline().split())

            # Allocate arrays
            self.kvecs = torch.zeros((self.nkpts, 3), dtype=ftype, device=device)
            self.weights = torch.zeros(self.nkpts, dtype=ftype, device=device)
            self.eig = torch.zeros((self.nspins, self.nkpts, self.nbands), dtype=ftype, device=device)
            self.occ = torch.zeros((self.nspins, self.nkpts, self.nbands), dtype=ftype, device=device)

            # Read k-points and band data
            for i in range(self.nkpts):
                f.readline()  # empty line
                line = list(map(float, f.readline().split()))
                self.kvecs[i, 0] = line[0]
                self.kvecs[i, 1] = line[1]
                self.kvecs[i, 2] = line[2]
                self.weights[i] = line[3]

                for j in range(self.nbands):
                    entries = list(map(float, f.readline().split()))
                    if self.nspins == 1:
                        self.eig[0, i, j], self.occ[0, i, j] = entries[1:3]
                    else:
                        self.eig[0, i, j], self.eig[1, i, j], self.occ[0, i, j], self.occ[1, i, j] = entries[1:5]

        # Identify the number of occupied bands for each spin channel
        self.noccs = self._find_number_of_occupied_bands()

    def _find_number_of_occupied_bands(self, threshold = 0.5) -> np.ndarray:
        """
        Identifies the number of occupied bands for each spin.

        Args:
            threshold (float): Occupation threshold to consider a band occupied.

        Returns:
            np.ndarray: Number of occupied bands per spin (nspins,).
        """
        noccs = np.zeros(self.nspins, dtype=np.int32)
        for ispin in range(self.nspins):
            occupied = np.where(self.occ[ispin, 0] >= threshold)[0] # first k point, all bands
            if occupied.size == 0:
                raise ValueError(f"No occupied bands found for spin {ispin}")
            noccs[ispin] = occupied.max() + 1
        return noccs
    
    def apply_scissors_correction(
        self, delta, stretch_cb: 1, stretch_vb: 1
    ) -> None:
        """
        Apply a scissor correction and optional stretching to conduction and valence bands.

        For conduction bands:
            E_cb = (E_cb - CBM) * stretch_cb + delta + CBM

        For valence bands:
            E_vb = (E_vb - VBM) * stretch_vb + VBM

        Parameters:
        ----------
        delta : float
            Scissor shift to apply to conduction band energies.
        stretch_cb : float, optional
            Stretching factor for conduction bands (default: 1.0).
        stretch_vb : float, optional
            Stretching factor for valence bands (default: 1.0).
        """
        assert hasattr(self, "eig") and self.eig.ndim == 3, "Expected eig to have shape (nspin, nk, nbands)"

        eig = self.eig

        for ispin in range(self.nspins):
            top_valence_idx = self.noccs[ispin] - 1
            bottom_conduction_idx = top_valence_idx + 1

            vbm = torch.max(eig[ispin, :, top_valence_idx])
            cbm = torch.min(eig[ispin, :, bottom_conduction_idx])

            # Apply to conduction bands
            eig[ispin, :, bottom_conduction_idx:] = (
                (eig[ispin, :, bottom_conduction_idx:] - cbm) * stretch_cb + cbm + delta
            )

            # Apply to valence bands
            eig[ispin, :, :bottom_conduction_idx] = (
                (eig[ispin, :, :bottom_conduction_idx] - vbm) * stretch_vb + vbm
            )

        self.eig = eig.to(ftype)

    def read_waveder(self, filename: str = 'WAVEDER') -> torch.Tensor:
        """
        Read position matrix elements (inter-band Berry connection) from a WAVEDER file.

        The WAVEDER file is generated when LOPTICS is set in VASP and contains the 
        off-diagonal matrix elements of the position operator:
            r[m, n, ik, isp, idir] = -<mk | r_idir | nk>

        These are interpreted as transition dipole matrix elements (-e * r).
        The diagonal matrix elements are set to be zeros. If LPEAD was set, the matrix
        elements between the bands within the conduction or valence subspaces are zeros.

        Parameters:
            filename (str): Path to the WAVEDER file. Defaults to 'WAVEDER'.

        Return:
            rmn (torch.Tensor): Array of shape (nspin, nkpts, nbands, ncder, 3)
                               containing position matrix elements.
        """
        try:
            with open(filename, "rb") as fp:
                nbands, ncder, nkpts, nspins = readData(fp, np.int32)

                if nbands != self.nbands:
                    raise ValueError(f"nbands mismatch: file={nbands}, expected={self.nbands}")
                if nkpts != self.nkpts:
                    raise ValueError(f"nkpts mismatch: file={nkpts}, expected={self.nkpts}")
                if nspins != self.nspins:
                    raise ValueError(f"nspins mismatch: file={nspins}, expected={self.nspins}")

                _ = readData(fp, np.float64)  # nodes_in_dielectric_function
                _ = readData(fp, np.float64)  # wplasmon

                if self.lgamma:
                    cder_flat_np = readData(fp, np.float64)
                    cder_flat = torch.from_numpy(cder_flat_np.copy()).to(device).to(ftype)
                else:
                    cder_flat_np = readData(fp, np.complex64)
                    cder_flat = torch.from_numpy(cder_flat_np.copy()).to(device).to(ctype)

                # Original shape: (3, nspins, nkpts, ncder, nbands)
                cder = cder_flat.reshape((3, nspins, nkpts, ncder, nbands))

                # Transpose to: (nspins, nkpts, nbands, ncder, 3)
                rmn = -cder.permute(1, 2, 4, 3, 0)

        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to read WAVEDER file '{filename}': {e}")
        return rmn
    
    def read_eigder(self, filename: str = 'EIGDER') -> torch.Tensor:
        """
        Read band velocities from a VASP-generated EIGDER file.
    
        The file contains the band gradients with respect to k: 
            v_nk = ∂E_nk / ∂k

        Parameters:
            filename (str): Path to the EIGDER file. Defaults to 'EIGDER'.

        Return:
            eig_der (torch.Tensor): Array of shape (nspins, nkpts, nbands, 3)
                               containing the band velocities.
        """
        try:
            with open(filename, "rb") as fp:
                nbands, nkpts, nspins = readData(fp, np.int32)

                if nbands != self.nbands:
                    raise ValueError(f"nbands mismatch: file={nbands}, expected={self.nbands}")
                if nkpts != self.nkpts:
                    raise ValueError(f"nkpts mismatch: file={nkpts}, expected={self.nkpts}")
                if nspins != self.nspins:
                    raise ValueError(f"nspins mismatch: file={nspins}, expected={self.nspins}")

                energy_der_flat_np = readData(fp, np.float64)
                energy_der_flat = torch.from_numpy(energy_der_flat_np.copy()).to(device).to(ftype)
            
                # Original shape: (3, nspins, nkpts, nbands)
                energy_der = energy_der_flat.reshape((3, nspins, nkpts, nbands))
            
                # Final shape: (nspins, nkpts, nbands, 3)
                eig_der = energy_der.permute(1,2,3,0)

        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to read EIGDER file '{filename}': {e}")

        return eig_der
    
    def calc_velocity_matrix_elements(self, rmn: torch.Tensor, eig_der: torch.Tensor) -> torch.Tensor:
        """
        Calculate the velocity matrix elements between Bloch states.

        The off-diagonal velocity matrix elements vmn are computed as:
            vmn = (i / ħ) * (E_mk - E_nk) * rmn
        where `rmn` are the position matrix elements, and `E_mk`, `E_nk` are band energies.
        Velocity matrix elements between degenerate states are basis dependent and
        are set to zero (handled externally if needed).
        The diagonal matrix elements are eig_der.

        Parameters
        ----------
        rmn : torch.Tensor
            Position matrix elements with shape (nspins, nkpts, nbands, nbands, 3).
        eig_der : torch.Tensor
            Band velocity with shape (nspins, nkpts, nbands, 3).

        Returns
        -------
        vmn : torch.Tensor
            Velocity matrix elements with the same shape as `rmn`.
        """
        # check the dimensions of rmn
        nbands, ncder = rmn.shape[2:4]
        if nbands != ncder:
            raise ValueError(f"nbands/ncder mismatch: file={ncder}, expected={nbands}, set LVEL=T")
            
        eig = self.eig  # shape: (nspins, nkpts, nbands)
        emn = eig[:, :, :, None] - eig[:, :, None, :]  # Shape: (nspins, nkpts, nbands, nbands)

        # vmn = i * (E_mk - E_nk) * rmn
        #vmn = 1j * np.einsum('ijklm,ijkl->ijklm', rmn, emn, optimize=True)
        vmn = 1j*rmn*emn[...,None]

        # Create an index array for the diagonal elements
        diagonal_indices = np.arange(self.nbands)

        # Assign the diagonal elements directly
        vmn[:, :, diagonal_indices, diagonal_indices, :] = eig_der[:, :, :, :].to(ctype)

        return vmn
    
    def calc_magnetic_dipole_electric_quadrupole(
        self,
        rmn: torch.Tensor,
        vmn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the magnetic dipole and electric quadrupole transition matrix elements.

        This function evaluates the linear-in-q part of the light-matter interaction matrix:
            ⟨mk | e^{iq·r} v + v e^{iq·r} | nk⟩ / 2 ≈ iq W_ij
        where:
            W_ij = (⟨mk | r_i v_j + v_j r_i | nk⟩) / 2

        The decomposition is:
            W_ij = M_ij + i·ω·Q_ij
        where:
            - M (antisymmetric part) corresponds to the magnetic dipole
            - Q (symmetric part) corresponds to the electric quadrupole (absorbing ω)

        Parameters
        ----------
        rmn : torch.Tensor
            Position matrix elements with shape (nspins, nk, nbands, nbands, 3).
        vmn : torch.Tensor
            Velocity matrix elements with the same shape as `rmn`.

        Returns
        -------
        M : torch.Tensor
            Magnetic dipole matrix elements with shape (nspins, nkpts, nc, nv, 3, 3).
        Q : torch.Tensor
            Electric quadrupole matrix elements with shape (nspins, nkpts, nc, nv, 3, 3).
        """
        bcb = self.noccs # shape (nspins,)
        c0 = bcb[0]
        if self.nspins == 1:
            r1 = rmn[:,:,c0:,:,:]
            v1 = vmn[:,:,:,:c0,:]
            v2 = vmn[:,:,c0:,:,:]
            r2 = rmn[:,:,:,:c0,:]
        else:
            r1 = rmn
            r2 = rmn
            v1 = vmn
            v2 = vmn
        
        # Compute W_ij = (r_i v_j + v_j r_i) / 2
        W = (
            torch.einsum('skmpi,skpnj->skmnij', r1, v1) +
            torch.einsum('skmpj,skpni->skmnij', v2, r2)
        ) / 2

        # Decompose into symmetric (Q) and antisymmetric (M) parts
        W_T = W.transpose(4,5)  # transpose last two indices
        
        Q = (W + W_T) / 2  # symmetric: electric quadrupole
        M = (W - W_T) / 2  # antisymmetric: magnetic dipole

        
        #antisym = (W - W_T) / 2  # antisymmetric: magnetic dipole

        # Magnetic dipole vector components (ε_ijk antisymmetric tensor contraction)
        #M = np.stack((
        #    antisym[..., 1, 2],  # M_x = W_yz - W_zy
        #    antisym[..., 2, 0],  # M_y = W_zx - W_xz
        #    antisym[..., 0, 1],  # M_z = W_xy - W_yx
        #), axis=-1)  # shape: (nspins, nk, nbands, nbands, 3)

        return M, Q
    
    def calc_natural_optical_activity(
        self,
        V: torch.Tensor,
        M: torch.Tensor,
        Q: torch.Tensor,
        vol, 
        emin=0,
        emax=10,
        ne=200,
        eta=0.1,
        rotations: torch.Tensor = None
    ) -> None:
        """
        Compute the natural optical activity tensor γ(ω).

        γ_ijl = - (4π e² / V ω² ℏ) { 
                  ∑_{cvk} [V_i* W_lj - V_j W_li*] * f(ω)
                + ∑_{cvk} V_i* V_j (v_c^l + v_v^l)/2 g(ω)}
        
        where:
            - f(ω) = 1/(ω - E_cv + iη) - 1/(ω + E_cv + iη)
            - g(ω) = 1/(ω - E_cv + iη)² + 1/(ω + E_cv + iη)²
            - OR + iCD = ω²/2c² * γ

        To remove the divergence of 1/ω², we replace \omega by \omega_{cvk} due to delta function.
        A correction term has to be added:
            (4π e² / V ℏ) ∑_{cvk} V_i* V_j (v_c^l + v_v^l)/\omega_{cvk}^3 * f(ω)
    
        Args:
            V (torch.Tensor): velocity matrix elements, shape (nspins, nkpts, nbands, nbands, 3)
            M (torch.Tensor): magnetic dipole matrix elements, shape (nspins, nkpts, nc, nv, 3, 3)
            Q (torch.Tensor): electric quadrupole matrix elements, shape (nspins, nkpts, nc, nv, 3, 3)
            vol (float): unit cell volume [A^3]
            emin (float): Minimum photon energy [eV]
            emax (float): Maximum photon energy [eV]
            ne (int): Number of energy grid points
            eta (float): Broadening parameter in energy denominator [eV]
            rotations (torch.Tensor, optional): Symmetry operations for symmetrization, shape (nr, 3, 3)
        """


        gamma_m = torch.zeros((ne, 3, 3), dtype=ctype, device=device)
        gamma_q = torch.zeros((ne, 3, 3), dtype=ctype, device=device)
        gamma_v = torch.zeros((ne, 3, 3), dtype=ctype, device=device)

        # average velocity vbar = (v_c + v_v ) / 2
        V_diag = torch.diagonal(V, dim1=2, dim2=3).transpose(2,3)
        vbar = (V_diag[:,:,:,None,:] + V_diag[:,:,None,:,:] ) / 2

        # energy difference
        eig = self.eig
        ec = eig[:,:,:,None]
        ev = eig[:,:,None,:]
        ecv = ec - ev

        # Slice the nbands to nc, nv
        bcb = self.noccs # shape (nspins,)
        c0 = bcb[0]
        if self.nspins == 1:
            V = V[:,:,c0:,:c0,:]
            vbar = vbar[:,:,c0:,:c0,:]
            ecv = ecv[:,:,c0:,:c0]

        # Compute antisymmetric combinations V* M and V* Q
        vm = torch.zeros_like(M, device=device)
        vq = torch.zeros_like(M, device=device)
        vv = torch.zeros_like(M, device=device)

        cyclic_idx = torch.tensor([[1, 2, 0], [2, 0, 1]], device=device)  # i, j pairs
        for l in range(3):
            i, j = cyclic_idx[0], cyclic_idx[1] # shape (3,)
            vm[...,l] = (V[...,i].conj()*M[...,l,j] - V[...,j]*M[...,l,i].conj())/ecv[...,None]**2
            vq[...,l] = (V[...,i].conj()*Q[...,l,j] - V[...,j]*Q[...,l,i].conj())/ecv[...,None]**2
            vv[...,l] = (V[...,i].conj()*V[...,j]*(vbar[...,l][...,None]))/ecv[...,None]**2

        vv_corr = -2*vv/ecv[...,None,None]

        wtk = self.weights[None,:,None,None,None,None]  # shape: (nkpts, 1, 1)
        vm = (vm*wtk).real.to(ctype)
        vq = (vq*wtk).real.to(ctype)
        vv = (vv*wtk).imag.to(ctype)
        vv_corr = (vv_corr*wtk).imag.to(ctype)

        omega = torch.linspace(emin, emax, ne, dtype=ftype, device=device)  # shape: (ne,)
        en = omega[:, None, None, None]    # shape: (ne, 1, 1, 1)
        for ik in range(self.nkpts):

            w1 = 1 / (en - ecv[:,ik,:,:] + 1j * eta)
            w2 = 1 / (en + ecv[:,ik,:,:] + 1j * eta)
                           
            f = w1 - w2  # shape: (ne, nspins, nc, nv)
            g = w1*w1 + w2*w2


            gamma_m += torch.tensordot(f, vm[:,ik,...], dims=([1,2,3], [0,1,2]))  
            gamma_q += torch.tensordot(f, vq[:,ik,...], dims=([1,2,3], [0,1,2])) 
            gamma_v += torch.tensordot(g, vv[:,ik,...], dims=([1,2,3], [0,1,2])) \
                    + torch.tensordot(f, vv_corr[:,ik,...], dims=([1,2,3], [0,1,2]))

        # Symmetrize if needed
        if rotations is not None:
            gamma_m = symmetrize_axial_tensor(gamma_m, rotations)
            gamma_q = symmetrize_axial_tensor(gamma_q, rotations)
            gamma_v = symmetrize_axial_tensor(gamma_v, rotations)

        # Conversion factor to degree/mm from atomic units
        prefactor = -4 * PI / vol * 2 / self.nspinors / 2 / C_LIGHT**2 / AU2EV
        prefactor *= 180 / PI / AU2A * 1e7

        gamma_m = gamma_m.reshape(ne, 9) * omega[:,None]**2 * prefactor
        gamma_q = gamma_q.reshape(ne, 9) * omega[:,None]**2 * prefactor
        gamma_v = gamma_v.reshape(ne, 9) * omega[:,None]**2 * prefactor
        gamma_total = gamma_m + gamma_q + gamma_v

        # Save and store
        _save_gamma(gamma_m, omega, 'm')
        _save_gamma(gamma_q, omega, 'q')
        _save_gamma(gamma_v, omega, 'v')
        _save_gamma(gamma_total, omega, 'tot')

def _save_gamma(gamma: torch.Tensor, omega: torch.Tensor, kernel: str) -> None:
    header = '   E(eV)        YZX            YZY            YZZ            ZXX            ' \
           +   'ZXY            ZXZ            XYX            XYY            XYZ'
    fmt = '%10.5f' + '%20.8f'*9
    omega_np = omega.cpu().numpy()[:, None]
    gamma_np = gamma.cpu().numpy()
    filr = np.concatenate((omega_np, gamma_np.real), axis=1)
    fili = np.concatenate((omega_np, gamma_np.imag), axis=1)
    np.savetxt('gamma_real_'+kernel+'.dat', filr, fmt=fmt, header=header)
    np.savetxt('gamma_imag_'+kernel+'.dat', fili, fmt=fmt, header=header)
