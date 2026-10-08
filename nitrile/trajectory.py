from pathlib import Path

import numpy as np


MASSES = {
    "H": 1.008, "D": 2.014, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904,
}


def topology_elements(atoms, type_elements=None):
    if type_elements is not None:
        missing = sorted(set(str(t) for t in atoms.types) - set(type_elements))
        if missing:
            raise ValueError(f"The atom-type element mapping is missing types {missing}")
        return tuple(type_elements[str(t)] for t in atoms.types)
    try:
        elements = tuple(str(e) for e in atoms.elements)
        if all(e.strip() for e in elements):
            return elements
    except Exception as error:

        if type(error).__name__ not in {"NoDataError", "AttributeError"}:
            raise
    try:
        masses = np.asarray(atoms.masses, dtype=np.float64)
    except Exception as error:
        raise ValueError("Topology has no elements or masses; provide a topology with element identities or standard atomic masses") from error
    elements = []
    for mass in masses:
        element = min(MASSES, key=lambda e: abs(MASSES[e] - mass))
        if not np.isfinite(mass) or abs(MASSES[element] - mass) > 0.25:
            raise ValueError(f"Cannot identify element from mass {mass}; provide a topology with element identities or standard atomic masses")
        elements.append(element)
    return tuple(elements)


def parse_indices(value):

    result = []
    for token in value.split(","):
        if ":" in token:
            fields = token.split(":")
            if len(fields) != 2:
                raise ValueError("Use inclusive atom ranges such as 1:6")
            first, last = (int(v) for v in fields)
            if first > last:
                raise ValueError("Atom ranges must be increasing")
            result.extend(range(first, last + 1))
        else:
            result.append(int(token))
    if not result or len(set(result)) != len(result):
        raise ValueError("Atom indices must be nonempty and unique")
    return result


class Trajectory:
    def __init__(self, topology, trajectory, *, engine="auto", type_elements=None,
                 lammps_units="real", timestep=None, atom_style=None):
        try:
            import MDAnalysis as mda
        except ImportError as error:
            raise ImportError('Install packages with pip install -r requirements.txt') from error
        self.topology = Path(topology)
        self.trajectory = Path(trajectory)
        if engine == "auto":
            engine = "gromacs" if self.topology.suffix.lower() == ".tpr" else "lammps"
        self.engine = engine
        self.step_ps = None
        if engine == "gromacs":
            if self.topology.suffix.lower() != ".tpr":
                raise ValueError("GROMACS input requires a .tpr topology containing charges")
            self.universe = mda.Universe(str(topology), str(trajectory), convert_units=True)
        elif engine == "lammps":
            if lammps_units not in ("real", "metal"):
                raise ValueError("LAMMPS units must be real or metal; dimensionless LJ units are unsupported")
            if timestep is None or not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("LAMMPS requires --timestep: fs for units real, ps for units metal")
            self.step_ps = float(timestep) * (0.001 if lammps_units == "real" else 1.0)
            kwargs = {"topology_format": "DATA", "convert_units": True}
            if atom_style is not None:
                kwargs["atom_style"] = atom_style
            if self.trajectory.suffix.lower() == ".dcd":
                kwargs.update(format="LAMMPS", timeunit="fs" if lammps_units == "real" else "ps",
                              lengthunit="Angstrom")
            else:
                kwargs.update(format="LAMMPSDUMP", dt=self.step_ps)
            self.universe = mda.Universe(str(topology), str(trajectory), **kwargs)
        else:
            raise ValueError("engine must be auto, gromacs or lammps")
        atoms = self.universe.atoms
        self.elements = topology_elements(atoms, type_elements)
        try:
            self.charges = np.asarray(atoms.charges, dtype=np.float64)
        except Exception as error:
            raise ValueError("Topology must contain partial charges; use a GROMACS TPR or LAMMPS full-style DATA file") from error
        try:
            self.molecule_ids = np.asarray(atoms.molnums, dtype=np.int64)
            self.molecule_source = "topology_molnums"
        except Exception:
            if self.engine == "lammps":
                self.molecule_ids = np.asarray(atoms.resids, dtype=np.int64)
                self.molecule_source = "LAMMPS_molecule_tags"
            else:
                self.molecule_ids = np.asarray(atoms.fragindices, dtype=np.int64)
                self.molecule_source = "topology_connected_fragments"
        try:
            self.bonds = np.asarray(atoms.bonds.indices, dtype=np.int64)
        except Exception:
            self.bonds = None
        self.atom_ids = np.asarray(atoms.ids, dtype=np.int64)

    def resolve_atom_numbers(self, numbers, *, index_base=1):
        if self.engine == "lammps":
            lookup = {int(number): i for i, number in enumerate(self.atom_ids)}
            if len(lookup) != len(self.atom_ids):
                raise ValueError("LAMMPS topology contains duplicate atom IDs")
            missing = set(numbers) - set(lookup)
            if missing:
                raise ValueError(f"LAMMPS atom IDs absent from topology: {sorted(missing)}")
            return np.array([lookup[number] for number in numbers], dtype=np.int64)
        indices = np.asarray(numbers, dtype=np.int64) - index_base
        if np.any(indices < 0) or np.any(indices >= len(self.atom_ids)):
            raise ValueError("GROMACS atom numbers are outside the topology")
        return indices

    def frames(self, *, start=0, stop=None, stride=1):
        if start < 0 or stride < 1 or (stop is not None and stop <= start):
            raise ValueError("Require start >= 0, stride >= 1 and stop > start")
        n_frames = len(self.universe.trajectory)
        if start >= n_frames:
            raise ValueError("Start frame is beyond the trajectory")
        stop = n_frames if stop is None else min(stop, n_frames)
        for index in range(start, stop, stride):
            ts = self.universe.trajectory[index]
            cell = None
            if ts.dimensions is not None and np.all(np.asarray(ts.dimensions)[:3] > 0):
                cell = np.asarray(ts.triclinic_dimensions, dtype=np.float64)
            time_ps = float(ts.time)
            if self.engine == "lammps" and "step" in ts.data:
                time_ps = float(ts.data["step"]) * self.step_ps
            yield index, time_ps, np.asarray(ts.positions, dtype=np.float64), cell
