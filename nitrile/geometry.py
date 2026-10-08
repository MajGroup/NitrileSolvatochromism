from dataclasses import dataclass
import itertools
import math

import numpy as np

from .network import ELEMENTS


@dataclass(frozen=True)
class Frame:
    global_x: np.ndarray
    element_index: np.ndarray
    continuous: np.ndarray
    atom_mask: np.ndarray
    aligned_combined_coordinates_A: np.ndarray
    environment_elements: tuple[str, ...]
    molecule_index: np.ndarray
    whole_tt6_response: float
    per_molecule_tt6_response: np.ndarray
    response_profile: str = "acetonitrile"
    prediction_scope: str = "acetonitrile_response"
    scaffold_correction_cm1: float = 0.0
    transfer_metadata: dict | None = None


def canonical_element(value):
    value = str(value).strip()
    if value == "D":
        return "H"
    result = value[:1].upper() + value[1:].lower()
    if not result or not result.isalpha() or len(result) > 2:
        raise ValueError(f"Invalid element identity: {value!r}")
    return result


def cell_matrix(box):
    if box is None:
        return None
    cell = np.asarray(box, dtype=np.float64)
    if cell.shape == (3,):
        cell = np.diag(cell)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("box must contain three lengths or three lattice vectors")
    if np.linalg.det(cell) <= 0:
        raise ValueError("box must have positive volume")
    return cell


def minimum_image(vectors, cell):
    vectors = np.asarray(vectors, dtype=np.float64)
    if cell is None:
        return vectors.copy()
    if np.allclose(cell, np.diag(np.diag(cell)), rtol=0, atol=1e-12):
        lengths = np.diag(cell)
        return vectors - np.rint(vectors / lengths) * lengths


    shape = vectors.shape
    flat = vectors.reshape(-1, 3)
    fractional = flat @ np.linalg.inv(cell)
    base = np.rint(fractional)
    best = np.empty_like(flat)
    norm = np.full(len(flat), np.inf)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        candidate = flat - (base + np.asarray(offset)) @ cell
        squared = np.einsum("ij,ij->i", candidate, candidate)
        take = squared < norm
        best[take] = candidate[take]
        norm[take] = squared[take]
    return best.reshape(shape)


def whole_and_image(coordinates, origin, cell, bonds=None):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if cell is None or not len(coordinates):
        return coordinates.copy()
    if bonds is None or len(bonds) == 0:
        whole = coordinates[0] + minimum_image(coordinates - coordinates[0], cell)
    else:
        neighbors = [[] for _ in coordinates]
        for first, second in bonds:
            neighbors[int(first)].append(int(second))
            neighbors[int(second)].append(int(first))
        whole = np.empty_like(coordinates)
        seen = {0}
        whole[0] = coordinates[0]
        pending = [0]
        while pending:
            first = pending.pop()
            for second in neighbors[first]:
                if second not in seen:
                    whole[second] = whole[first] + minimum_image(
                        coordinates[second] - coordinates[first], cell
                    )
                    seen.add(second)
                    pending.append(second)
        if len(seen) != len(coordinates):
            raise ValueError("Molecule connectivity is disconnected; check molecule IDs")
    center = whole.mean(axis=0)
    return whole + minimum_image(center - origin, cell) - (center - origin)


def align_reference(reference, solute, *, all_atoms=False):

    source, target = (reference, solute) if all_atoms else (reference[[0, 4, 5]], solute[[0, 4, 5]])
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, target_center - source_center @ rotation


def electrostatic_features(environment, charges, reference, physics):
    c_xyz, n_xyz = reference[4], reference[5]
    axis = n_xyz - c_xyz
    length = np.linalg.norm(axis)
    axis = axis / length
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, trial)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    midpoint = 0.5 * (c_xyz + n_xyz)
    sites = [c_xyz, n_xyz, n_xyz + physics["lone_pair_distance_A"] * axis]
    for i in range(physics["ring_n_sites"]):
        angle = 2 * math.pi * i / physics["ring_n_sites"]
        sites.append(midpoint + physics["ring_radius_A"] * (
            math.cos(angle) * u + math.sin(angle) * v
        ))
    potentials, parallel_fields = [], []
    for site in sites:
        dr = site[None, :] - environment
        distance = np.maximum(np.linalg.norm(dr, axis=1), 1e-8)
        potentials.append(float(np.sum(charges / distance)))
        field = np.sum(charges[:, None] * dr / distance[:, None]**3, axis=0)
        parallel_fields.append(float(np.dot(field, axis)))
    return np.array([
        potentials[0] - potentials[1],
        potentials[2] - potentials[1],
        float(np.mean(potentials[3:])) - potentials[1],
        parallel_fields[1],
        (parallel_fields[1] - parallel_fields[0]) / length,
    ], dtype=np.float64)


def tt6_response(environment, elements, molecule_index, physics):
    states = physics["solute_states_A"]
    alpha = physics["free_atom_alpha_A3"]
    solute_alpha = np.array([alpha[e] for e in ("C", "H", "H", "H", "C", "N")])
    missing = sorted(set(elements) - set(alpha))
    if missing:
        raise ValueError(f"No frozen dispersion constant for elements {missing}")
    environment_alpha = np.array([alpha[e] for e in elements])
    weights = np.sqrt(solute_alpha[:, None] * environment_alpha[None, :])
    n_molecules = int(molecule_index.max()) + 1 if len(molecule_index) else 0
    per_state, whole_state = [], []
    for solute in states:
        distance = np.maximum(np.linalg.norm(
            solute[:, None, :] - environment[None, :, :], axis=2
        ), 1e-8)
        x = physics["tt_beta_bohr_inv"] * physics["bohr_per_A"] * distance
        series = np.zeros_like(x)
        for k in range(7):
            series += x**k / math.factorial(k)
        damping = np.where(x > 1e-12, 1 - np.exp(-x) * series, 0)
        pair_energy = weights * damping / distance**6
        whole_state.append(float(-np.sum(pair_energy)))
        atom_energy = -np.sum(pair_energy, axis=0)
        per_state.append(np.bincount(molecule_index, weights=atom_energy,
                                     minlength=n_molecules))
    step, coefficient = physics["q_step"], physics["response_coefficient"]
    per_state = np.array(per_state)
    # Combine the direct curvature and mechanical-anharmonicity responses.
    per_molecule = coefficient * (
        (per_state[1] - 2 * per_state[0] + per_state[2]) / step**2
        - (per_state[3] - per_state[4]) / (2 * step)
    )
    direct = coefficient * (whole_state[1] - 2 * whole_state[0] + whole_state[2]) / step**2
    mechanical = -coefficient * (whole_state[3] - whole_state[4]) / (2 * step)
    whole = float(direct + mechanical)
    if not np.isclose(per_molecule.sum(), whole, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Per-molecule dispersion does not close to the complete response")
    return whole, per_molecule


def prepare_frame(positions, elements, charges, molecule_ids, solute_indices,
                  nitrile_indices, physics, *, box=None, bonds=None,
                  global_electrostatics=None, virtual_pose=False):
    positions = np.asarray(positions, dtype=np.float64)
    elements = tuple(canonical_element(e) for e in elements)
    charges = np.asarray(charges, dtype=np.float64)
    molecules = np.asarray(molecule_ids)
    solute_indices = np.asarray(solute_indices)
    nitrile_indices = np.asarray(nitrile_indices)
    n_atoms = len(elements)
    if (positions.shape != (n_atoms, 3) or charges.shape != (n_atoms,)
            or molecules.shape != (n_atoms,) or not np.isfinite(positions).all()
            or not np.isfinite(charges).all()):
        raise ValueError("Positions, elements, charges and molecule IDs must cover the same atoms")
    if molecules.dtype.kind not in "iu" or np.any(molecules < 0):
        raise ValueError("Molecule IDs must be nonnegative integers")
    for indices in (solute_indices, nitrile_indices):
        if (indices.dtype.kind not in "iu" or indices.ndim != 1
                or np.any(indices < 0) or np.any(indices >= n_atoms)
                or len(set(indices.tolist())) != len(indices)):
            raise ValueError("Atom indices must be unique, zero-based integers within the frame")
    if len(nitrile_indices) != 3 or not set(nitrile_indices).issubset(set(solute_indices)):
        raise ValueError("nitrile_indices must be the solute's C(methyl), C(nitrile), N indices")
    if tuple(elements[i] for i in nitrile_indices) != ("C", "C", "N"):
        raise ValueError("The ordered nitrile atoms must have elements C, C, N")
    hydrogens = [i for i in solute_indices if elements[i] == "H"]
    if len(solute_indices) != 6 or len(hydrogens) != 3:
        raise ValueError("The acetonitrile model requires all six CH3CN/CD3CN atoms. Use --probe auto for a larger solute.")
    order = np.array([nitrile_indices[0], *hydrogens, nitrile_indices[1], nitrile_indices[2]])
    cell = cell_matrix(box)
    all_bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2) if bonds is not None else None
    if all_bonds is not None and (np.any(all_bonds < 0) or np.any(all_bonds >= n_atoms)):
        raise ValueError("Bond indices are outside the frame")
    bond_neighbors = [[] for _ in range(n_atoms)]
    if all_bonds is not None:
        for first, second in all_bonds:
            bond_neighbors[int(first)].append(int(second))
            bond_neighbors[int(second)].append(int(first))

    def local_bonds(indices):
        if all_bonds is None:
            return None
        lookup = {int(i): j for j, i in enumerate(indices)}
        return [(lookup[first], lookup[second]) for first in lookup
                for second in bond_neighbors[first] if second in lookup and first < second]

    solute = whole_and_image(positions[order], positions[order].mean(axis=0), cell,
                            local_bonds(order))
    bond = np.linalg.norm(solute[5] - solute[4])
    if not 0.85 <= bond <= 1.45:
        raise ValueError(f"C≡N distance {bond:.3f} Angstrom is outside the calibrated interval; check indices and units")
    midpoint = 0.5 * (solute[4] + solute[5])
    selected = set(solute_indices.tolist())
    groups = {}
    for index in range(n_atoms):
        if index not in selected:
            groups.setdefault(int(molecules[index]), [])
    for index, molecule in enumerate(molecules):
        if int(molecule) in groups:
            groups[int(molecule)].append(index)
    environment_coordinates, environment_elements, environment_charges, partition = [], [], [], []
    for molecule_number, group in enumerate(groups):
        indices = np.asarray(groups[group], dtype=np.int64)
        if selected.intersection(indices.tolist()):
            raise ValueError("Selected solute cuts through a molecule; select its complete atom group")
        if global_electrostatics is None and abs(charges[indices].sum()) > 5e-4:
            raise ValueError("The default electrostatic calculation requires neutral environment molecules. Charged environments need externally supplied electrostatic features through the Python API.")
        whole = whole_and_image(positions[indices], midpoint, cell, local_bonds(indices))
        environment_coordinates.extend(whole - midpoint)
        environment_elements.extend(elements[i] for i in indices)
        environment_charges.extend(charges[indices])
        partition.extend([molecule_number] * len(indices))
    environment = np.asarray(environment_coordinates, dtype=np.float64).reshape(-1, 3)
    solute = solute - midpoint
    reference = physics["reference_coordinates_A"]
    rotation, translation = align_reference(reference, solute, all_atoms=virtual_pose)
    reference_midpoint = 0.5 * (reference[4] + reference[5])
    aligned = (environment - translation) @ rotation.T - reference_midpoint
    centered_reference = reference - reference_midpoint
    partition = np.asarray(partition, dtype=np.int64)
    environment_charges = np.asarray(environment_charges, dtype=np.float64)
    global_x = electrostatic_features(aligned, environment_charges, centered_reference, physics)
    if global_electrostatics is not None:
        global_x = np.asarray(global_electrostatics, dtype=np.float64)
        if global_x.shape != (5,) or not np.isfinite(global_x).all():
            raise ValueError("global_electrostatics must be the five finite frozen-map observables")
    n_molecules = len(groups)
    max_atoms = int(np.bincount(partition).max()) if n_molecules else 1
    identity = np.zeros((n_molecules, max_atoms), dtype=np.int64)
    continuous = np.zeros((n_molecules, max_atoms, 5), dtype=np.float32)
    mask = np.zeros((n_molecules, max_atoms), dtype=np.float32)
    c_xyz, n_xyz = centered_reference[4], centered_reference[5]
    axis = n_xyz - c_xyz
    axis = axis / np.linalg.norm(axis)
    midpoint = 0.5 * (c_xyz + n_xyz)
    for molecule in range(n_molecules):
        for atom, index in enumerate(np.flatnonzero(partition == molecule)):
            xyz = aligned[index]
            vector = xyz - midpoint
            axial = float(np.dot(vector, axis))
            radial = math.sqrt(max(float(np.dot(vector, vector)) - axial**2, 0))
            element = environment_elements[index]
            identity[molecule, atom] = ELEMENTS.index(element) if element in ELEMENTS else ELEMENTS.index("OTHER")
            continuous[molecule, atom] = (0, np.linalg.norm(xyz - c_xyz),
                                         np.linalg.norm(xyz - n_xyz), axial, radial)
            mask[molecule, atom] = 1
    whole_tt6, per_molecule_tt6 = tt6_response(aligned, environment_elements, partition, physics)
    return Frame(global_x, identity, continuous, mask,
                 np.vstack((centered_reference, aligned)), tuple(environment_elements),
                 partition, whole_tt6, per_molecule_tt6)
