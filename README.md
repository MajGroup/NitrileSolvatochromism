# Nitrile Solvatochromism

This program uses a trained neural-network model to predict how the surrounding molecular environment affects the C≡N stretching frequency of nitriles. It reads molecular dynamics (MD) trajectories from **GROMACS or LAMMPS** and calculates the instantaneous frequency shift at every saved frame.

Besides the total frequency shift, the model provides separate contributions from electrostatics, exchange repulsion, HF delocalization, and electron correlation. This makes it possible to follow how different intermolecular interactions affect the nitrile vibration as the solvent molecules move around it.

The trained model is included in `nitrile_model.pth`. For installation instructions, see [INSTALL.md](INSTALL.md).

## Running the model

### GROMACS

For a GROMACS trajectory, run:

```bash
python predict.py --topology run.tpr --trajectory trajectory.xtc --solute 1:6 --nitrile 1 5 6 --output shifts.txt
```

The `.tpr` file provides the molecular topology, atom types, and partial charges, while the `.xtc` file contains the trajectory coordinates.

### LAMMPS

For a LAMMPS trajectory, use:

```bash
python predict.py --topology system.data --trajectory trajectory.dump --solute 1:6 --nitrile 1 5 6 --timestep 0.1 --output shifts.txt
```

LAMMPS requires a full-style DATA file containing atomic charges and molecule IDs. The timestep is given in femtoseconds for `real` units. For `metal` units, add `--units metal` and specify the timestep in picoseconds.

### Selecting the nitrile group

In both examples, replace the atom numbers with those of your system.

- `--solute` selects the entire solute molecule, including all its atoms.
- `--nitrile` specifies three atoms in this order: the carbon attached to the nitrile group, the nitrile carbon, and the nitrile nitrogen.
- `--output` specifies where the calculated frequency shifts will be saved.

GROMACS atom numbering starts at 1. For LAMMPS, use the atom IDs from the DATA file.

## Reading the results

The program saves the results in a text file, with one row for each trajectory frame. Each row contains the simulation time, the individual interaction contributions, and the total C≡N frequency shift in cm⁻¹.

The electrostatic contribution is averaged across three trained models. The final frequency shift is selected as the median of their SCS-MP2 predictions, with the other interaction contributions taken from the corresponding model.

You can easily load the results into Python:

```python
import numpy as np

data = np.loadtxt("shifts.txt", ndmin=2)

time = data[:, 1]
total_shift = data[:, -1]
```

For longer trajectories, `--stride 10` processes every tenth frame. You can also select a range of frames with `--start` and `--stop`, or use `--overwrite` to replace an existing output file.

## Using the model with other molecules

The model was originally developed for acetonitrile using the CD₃CN vibrational response. Both CH₃CN and CD₃CN coordinate topologies are accepted, but the same CD₃CN vibrational parameters are used in both cases.

We also provide a correction for the nitrile group in the capped β-cyanoalanine model peptide (Ac–Bca–NHMe). The program recognizes this molecule automatically and applies the additional correction when the geometry is within the supported range of the two reference conformations.

For other nitrile-containing molecules, the acetonitrile model can be used as an approximate starting point by selecting `--probe virtual-acn`. These predictions use the original acetonitrile vibrational response without a molecule-specific correction.

The default electrostatic calculation assumes electrically neutral solvent molecules and uses their nearest periodic images.

## Trying the examples

Example trajectories are included to help you check that everything is working. The water examples contain three MD frames in GROMACS and LAMMPS formats.

You can also try the capped β-cyanoalanine example:

```bash
python predict.py --topology examples/bca.data --trajectory examples/bca.dump --solute 1:23 --nitrile 6 7 8 --timestep 0.1 --output bca.txt
```

The expected results are provided in `examples/expected_water.txt` and `examples/expected_bca.txt`.

The package includes the trained model and everything needed to calculate frequency shifts along MD trajectories. The original training data, fitting scripts, and routines for simulating linear and 2D IR spectra are not included in this release.
