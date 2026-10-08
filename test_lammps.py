from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from predict import run


FOLDER = Path(__file__).resolve().parent


def test_lammps():
    with TemporaryDirectory() as directory:
        output = run(
            topology=FOLDER / "examples/water.data",
            trajectory=FOLDER / "examples/water.dump",
            solute="1:6", nitrile=[1, 5, 6], timestep=0.1,
            output=Path(directory) / "shifts.txt",
        )
        data = np.loadtxt(output, ndmin=2)
        expected = np.loadtxt(FOLDER / "examples/expected_water.txt", ndmin=2)
        np.testing.assert_equal(data.shape, expected.shape)
        np.testing.assert_allclose(data[:, :2], expected[:, :2], atol=1e-8, rtol=0)
        # Allow small differences between float32 kernels on different machines.
        np.testing.assert_allclose(data[:, 2:], expected[:, 2:], atol=1e-4, rtol=0)
        np.testing.assert_allclose(data[:, 2:7].sum(axis=1), data[:, 7], atol=1e-8, rtol=0)


if __name__ == "__main__":
    test_lammps()
    print("LAMMPS example passed")
