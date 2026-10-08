# Install

Use Python 3.10–3.12. Open a terminal in this folder and run:

```sh
python -m pip install -r requirements.txt
python test_gromacs.py
python test_lammps.py
python test_bca.py
```

The tests compare every output contribution with the supplied reference files
and check that the contributions add to the total. Temporary output files are
removed when each test finishes. GROMACS and LAMMPS executables are not needed
to read trajectories.

The prediction checks use an absolute tolerance of 0.0001 cm⁻¹ to allow small
float32 differences between software versions and machines. Frame numbers,
times and the sum of contributions are checked separately with tighter tolerances.
