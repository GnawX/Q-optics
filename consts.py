import torch

ftype = torch.float32 
ctype = torch.complex64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PI      = torch.tensor(torch.pi, dtype=ftype)
C_LIGHT = torch.tensor(137.035999084, dtype=ftype)  # speed of light
AU2EV   = torch.tensor(27.211386245988, dtype=ftype)  # Hartree to eV
AU2A    = torch.tensor(0.52917721092, dtype=ftype)     # Bohr to Angstrom
