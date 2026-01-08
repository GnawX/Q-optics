# adaptive k-mesh refinement
from phonopy.interface.calculator import read_crystal_structure
import numpy as np
import spglib
import sys

# BZ (-1/2, 1/2]

mesh    = [24, 24, 18]  # base k mesh to be refined
meshf   = [216, 216, 162] # target k mesh 
shift   = [0, 0, 0]
tiny    = 0.000

# define the volume to be refined
amin, amax = 58/216 + tiny, 86/216 - tiny
bmin, bmax = 58/216 + tiny, 86/216 - tiny
cmin, cmax = 67/162 + tiny, -67/162 - tiny

atoms, _ = read_crystal_structure('POSCAR')

#coarse grid
mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, atoms, shift)
kpt = grid / np.array(mesh, dtype=float)
nkpts = len(kpt)
#kpt, counts = np.unique(mapping,return_counts=True)
#kpt = grid[kpt] / np.array(mesh, dtype=float)

ncoarse = 0
for i in range(nkpts):
    if ((amin < kpt[i,0] < amax and bmin < kpt[i,1] < bmax) \
        or (-amax < kpt[i,0] < -amin and -bmax < kpt[i,1] < -bmin)) \
        and (cmin < kpt[i,2] or kpt[i,2] < cmax):
       
        ncoarse += 1
 #       print(kpt[i,:])

#sys.exit()
#print('fine')
#fine grid
mapping, grid = spglib.get_ir_reciprocal_mesh(meshf, atoms, shift)
kptf = grid / np.array(meshf, dtype=float)
nkptsf = len(kptf)
#kptf, countsf = np.unique(mapping,return_counts=True)
#kptf = grid[kptf] / np.array(meshf, dtype=float)

indx = []
nfine = 0
for i in range(nkptsf):
    if ((amin < kptf[i,0] < amax and bmin < kptf[i,1] < bmax) \
        or (-amax < kptf[i,0] < -amin and -bmax < kptf[i,1] < -bmin)) \
        and (cmin < kptf[i,2] or kptf[i,2] < cmax):

        indx.append(i)
        nfine += 1
        #print(kptf[i,:])

print(ncoarse/np.product(mesh))
print(nfine/np.product(meshf))
with open('KPOINTS.DAT','w') as f:
     f.write('automatically kmesh \n')
     f.write('{:10d}\n'.format(nkpts-ncoarse+len(indx)))
     f.write('Reciprocal lattice \n')
     for i in range(nkpts):
         if ((amin < kpt[i,0] < amax and bmin < kpt[i,1] < bmax) \
             or (-amax < kpt[i,0] < -amin and -bmax < kpt[i,1] < -bmin)) \
             and (cmin < kpt[i,2] or kpt[i,2] < cmax):
             pass 
         else:
             f.write(('{:20.14f}'*3 +'{:20.14f}\n').format(*kpt[i,:],1))
     for i in indx:
         f.write(('{:20.14f}'*3 +'{:20.14f}\n').format(*kptf[i,:],1/nfine*ncoarse))
