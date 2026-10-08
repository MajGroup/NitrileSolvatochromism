# Nitrile solvatochromism

Read an MD trajectory and save the interaction contributions and total nitrile
frequency shift to a text file. All contributions are frequency shifts in cm⁻¹.
The trained model and fitted corrections are in `nitrile_model.pth`.
Installation and example tests are described in [INSTALL.md](INSTALL.md).

The ACN baseline uses a fixed CD₃CN vibrational response. CH₃CN and CD₃CN
coordinate topologies are accepted. Changing masses or isotope labels in the
topology does not select a different vibrational response.

## GROMACS

```sh
python predict.py --topology run.tpr --trajectory trajectory.xtc --solute 1:6 --nitrile 1 5 6 --output shifts.txt
```

The TPR supplies charges, elements and connectivity. The trajectory supplies
coordinates and box dimensions.

## LAMMPS

```sh
python predict.py --topology system.data --trajectory trajectory.dump --solute 1:6 --nitrile 1 5 6 --timestep 0.1 --output shifts.txt
```

Use a full-style DATA file with charges and molecule IDs. For `real` units,
give the timestep in fs. For `metal` units, add `--units metal` and give it in ps.

Select the **whole solute**. The three nitrile atoms are the attached carbon,
nitrile carbon and nitrogen, in that order. GROMACS numbers start at 1;
LAMMPS uses the atom IDs in the DATA file. Replace the example indices with
your own.

## Output

Every run writes these columns:

```text
frame time_ps electrostatic exchange_repulsion orbital_relaxation dispersion_correlation scaffold_correction total
```

The five contribution columns add to the total. HF corrections are included
in orbital relaxation; MP2 correlation is included in dispersion/correlation.
SCS-MP2 uses opposite-spin and same-spin weights of 1.2 and 1/3.
Each row ends with a comment identifying the response used.

The electrostatic contribution is averaged across three models. The total is
the median of three SCS-MP2 totals, and the remaining interaction contributions
come from the same selected member. The scaffold correction is then added.

To read the numerical columns, including a file containing a single frame:

```python
import numpy as np
data = np.loadtxt("shifts.txt", ndmin=2)
time = data[:, 1]
total = data[:, 7]
```

`--stride 10` reads every tenth frame. `--start` and `--stop` select a frame
range; the stop frame is excluded. `--overwrite` replaces an existing output.

## Larger solutes

The studied capped BCA molecule is recognised automatically. Its native HF
correction appears in `scaffold_correction`. The ACN opposite-spin and same-spin
correlation contributions are retained. The two supported reference conformers
are selected by geometry. Frames more than 0.10 Å heavy-atom RMSD from both
are rejected. This distance is a software acceptance threshold; it has no
calibrated relationship to prediction error. The BCA contact correction
supports environment elements H, C, N, O and S.

Other scaffolds use the transferred ACN response. They have no native
correction in this package. `--probe virtual-acn` explicitly selects that
baseline. The backbone is excluded from the solvent field when it is part
of the solute selection. The default electrostatic calculation requires
neutral environment molecules and uses their nearest periodic images.

## Examples

The water examples contain three MD frames, encoded in both formats.
The BCA example contains two frozen test structures:

```sh
python predict.py --topology examples/bca.data --trajectory examples/bca.dump --solute 1:23 --nitrile 6 7 8 --timestep 0.1 --output bca.txt
```

Reference outputs are in `examples/expected_water.txt` and
`examples/expected_bca.txt`.

This package contains the trained model and trajectory-inference scripts.
Training data, fitting scripts and IR/2D-IR spectral calculations are outside
this release.
