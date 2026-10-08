from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from predict import run


FOLDER = Path(__file__).resolve().parent


def test_bca():
    with TemporaryDirectory() as directory:
        output = run(
            topology=FOLDER / "examples/bca.data",
            trajectory=FOLDER / "examples/bca.dump",
            solute="1:23", nitrile=[6, 7, 8], timestep=0.1,
            output=Path(directory) / "shifts.txt",
        )
        data = np.loadtxt(output, ndmin=2)
        expected = np.loadtxt(FOLDER / "examples/expected_bca.txt", ndmin=2)
        np.testing.assert_equal(data.shape, expected.shape)
        np.testing.assert_allclose(data[:, :2], expected[:, :2], atol=1e-8, rtol=0)
        # Allow small differences between float32 kernels on different machines.
        np.testing.assert_allclose(data[:, 2:], expected[:, 2:], atol=1e-4, rtol=0)
        np.testing.assert_allclose(data[:, 2:7].sum(axis=1), data[:, 7], atol=1e-8, rtol=0)
        profiles = [
            line.partition("#")[2].strip()
            for line in output.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        np.testing.assert_equal(profiles, ["capped-bca A", "capped-bca B"])


if __name__ == "__main__":
    test_bca()
    print("BCA example passed")
