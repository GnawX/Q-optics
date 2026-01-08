# Setup for VASP calculations.

- Apply the patch to generate `EIGDER`, `EXCEIG`, `EXCWAV` files.
- Compile VASP with `-Dsingle_prec_bse`.
- Set `LVEL = T` for `LOPTICS` calculations.
- For excitonic effects, one can bypass the GW calculations using model BSE `LMODELHF`.

