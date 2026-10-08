import argparse
from pathlib import Path

import torch

from nitrile import NitrileModel
from nitrile.trajectory import Trajectory, parse_indices


FOLDER = Path(__file__).resolve().parent
TERMS = ["electrostatic_cm1", "exchange_repulsion_cm1", "orbital_relaxation_cm1",
         "dispersion_correlation_cm1", "scaffold_correction_cm1", "shift_cm1"]


def run(topology, trajectory, solute, nitrile, output="shifts.txt", engine="auto",
        timestep=None, units="real", probe="auto", start=0,
        stop=None, stride=1, model_file=None, threads=1, overwrite=False):
    output = Path(output)
    temporary = output.with_name(output.name + ".tmp")
    if not overwrite and (output.exists() or temporary.exists()):
        raise FileExistsError(f"{output} already exists; use another name or --overwrite")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    model = NitrileModel(model_file or FOLDER / "nitrile_model.pth")
    data = Trajectory(topology, trajectory, engine=engine, timestep=timestep,
                      lammps_units=units)
    solute = data.resolve_atom_numbers(parse_indices(solute))
    nitrile = data.resolve_atom_numbers(nitrile)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write("# Shift contributions in cm^-1; time in ps\n")
            file.write("# frame time_ps electrostatic exchange_repulsion orbital_relaxation "
                       "dispersion_correlation scaffold_correction total\n")
            for frame, time, positions, box in data.frames(start=start, stop=stop, stride=stride):
                result = model.predict(
                    positions, data.elements, data.charges, data.molecule_ids,
                    solute, nitrile, box=box, bonds=data.bonds,
                    response_profile=probe,
                )
                values = "\t".join(f"{result[term]:.10f}" for term in TERMS)
                note = result["response_profile"]
                if result["conformer"]:
                    note += " " + result["conformer"]
                file.write(f"{frame}\t{time:.8f}\t{values}\t# {note}\n")
                count += 1
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {count} frames to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Nitrile shifts from an MD trajectory")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--solute", required=True, help="Complete solute, e.g. 1:6 or 1:6,10:15")
    parser.add_argument("--nitrile", type=int, nargs=3, required=True,
                        help="Attached C, nitrile C, N (one-based atom numbers)")
    parser.add_argument("--output", default="shifts.txt")
    parser.add_argument("--engine", choices=["auto", "gromacs", "lammps"], default="auto")
    parser.add_argument("--timestep", type=float, help="LAMMPS: fs for real units, ps for metal")
    parser.add_argument("--units", choices=["real", "metal"], default="real")
    parser.add_argument("--probe", choices=["auto", "acetonitrile", "bca", "virtual-acn"], default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model", dest="model_file")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        run(**vars(args))
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
