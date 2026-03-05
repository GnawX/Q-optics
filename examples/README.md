# Setup for VASP calculations.

- Apply the patch to generate `EIGDER`, `EXCEIG`, `EXCWAV` files.
- Set `LVEL = T` for `LOPTICS` calculations.
  
For BSE calculations:
- Compile VASP with `-Dsingle_prec_bse` and `-DscaLAPACK`.
- One can bypass the GW-BSE calculations using model BSE `LMODELHF`.





