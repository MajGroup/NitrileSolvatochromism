from collections import Counter
from dataclasses import replace

import numpy as np

from .geometry import canonical_element, cell_matrix, prepare_frame, whole_and_image


def _rigid_fit(reference, target):
    center_r, center_t = reference.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((reference-center_r).T @ (target-center_t))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = center_t-center_r @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum(
        (reference @ rotation+translation-target)**2, axis=1))))
    return rotation, translation, rmsd


def _selected_system(positions, elements, charges, molecule_ids, solute_indices,
                     nitrile_indices, box, bonds):
    r = np.asarray(positions, dtype=np.float64)
    elements = tuple(canonical_element(e) for e in elements)
    q, molecules = np.asarray(charges, dtype=np.float64), np.asarray(molecule_ids)
    n = len(elements)
    if (r.shape != (n, 3) or q.shape != (n,) or molecules.shape != (n,)
            or not np.isfinite(r).all() or not np.isfinite(q).all()):
        raise ValueError("Positions, elements, charges and molecule IDs must cover the same atoms")
    if molecules.dtype.kind not in "iu" or np.any(molecules < 0):
        raise ValueError("Molecule IDs must be nonnegative integers")
    solute, nitrile = np.asarray(solute_indices), np.asarray(nitrile_indices)
    for indices in (solute, nitrile):
        if (indices.dtype.kind not in "iu" or indices.ndim != 1
                or not len(indices) or np.any(indices < 0) or np.any(indices >= n)
                or len(set(indices.tolist())) != len(indices)):
            raise ValueError("Atom indices must be unique, zero-based integers within the frame")
    if len(nitrile) != 3 or not set(nitrile).issubset(set(solute)):
        raise ValueError("nitrile_indices must identify three atoms in the complete selected solute")
    if tuple(elements[i] for i in nitrile) != ("C", "C", "N"):
        raise ValueError("The ordered nitrile atoms must have elements C, C, N")
    cell = cell_matrix(box)
    selected = set(map(int, solute))
    for group in np.unique(molecules[solute]):
        if any(int(i) not in selected for i in np.flatnonzero(molecules == group)):
            raise ValueError("Selected solute cuts through a molecule; select its complete atom group")
    all_bonds = None if bonds is None else np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    if all_bonds is not None and (np.any(all_bonds < 0) or np.any(all_bonds >= n)):
        raise ValueError("Bond indices are outside the frame")
    if all_bonds is not None:
        for first, second in all_bonds:
            if (int(first) in selected) != (int(second) in selected):
                raise ValueError("Selected solute cuts a covalent bond; include the complete scaffold")
    lookup = {int(i): j for j, i in enumerate(solute)}
    local_bonds = None if all_bonds is None else np.asarray([
        (lookup[int(i)], lookup[int(j)]) for i, j in all_bonds
        if int(i) in lookup and int(j) in lookup], dtype=np.int64).reshape(-1, 2)
    native = whole_and_image(r[solute], r[nitrile[1]], cell, local_bonds)
    local_nitrile = np.array([lookup[int(i)] for i in nitrile])
    cn_length = np.linalg.norm(native[local_nitrile[2]]-native[local_nitrile[1]])
    if not 0.85 <= cn_length <= 1.45:
        raise ValueError(f"C≡N distance {cn_length:.3f} Angstrom is outside the calibrated interval; check indices and units")
    environment_indices = np.array([i for i in range(n) if i not in selected], dtype=int)
    return (native, tuple(elements[i] for i in solute), local_bonds, local_nitrile,
            r, elements, q, molecules, environment_indices, all_bonds, cell)


def _baseline(virtual, coordinates, elements, q, molecules, indices, bonds,
              cell, physics, global_electrostatics):

    remap = {int(i): j+6 for j, i in enumerate(indices)}
    environment_bonds = [] if bonds is None else [
        (remap[int(i)], remap[int(j)]) for i, j in bonds
        if int(i) in remap and int(j) in remap]
    if bonds is not None:
        environment_bonds = np.asarray(
            [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)]+environment_bonds,
            dtype=np.int64).reshape(-1, 2)
    else:
        environment_bonds = None
    _, groups = np.unique(molecules[indices], return_inverse=True)
    return prepare_frame(
        np.vstack((virtual, coordinates[indices])),
        ("C", "H", "H", "H", "C", "N")+tuple(elements[i] for i in indices),
        np.r_[np.zeros(6), q[indices]], np.r_[np.zeros(6, dtype=int), groups+1],
        np.arange(6), [0, 4, 5], physics, box=cell, bonds=environment_bonds,
        global_electrostatics=global_electrostatics, virtual_pose=True)


def _virtual_geometry(native, nitrile, physics):
    neighbor, c, n = native[nitrile]
    z = n-c
    z /= np.linalg.norm(z)
    transverse = neighbor-c-((neighbor-c) @ z)*z
    if np.linalg.norm(transverse) < 1e-8:
        trial = np.eye(3)[np.argmin(np.abs(z))]
        transverse = trial-(trial @ z)*z
    x = transverse/np.linalg.norm(transverse)
    y = np.cross(z, x)
    reference = physics["reference_coordinates_A"]
    rz = reference[5]-reference[4]
    rz /= np.linalg.norm(rz)
    rx = reference[1]-reference[0]
    rx -= (rx @ rz)*rz
    rx /= np.linalg.norm(rx)
    ry = np.cross(rz, rx)
    rotation = np.column_stack((rx, ry, rz)) @ np.column_stack((x, y, z)).T
    return (reference-reference[4]) @ rotation+c


def _heavy_mappings(elements, bonds, nitrile, profile):

    expected = tuple(profile["elements"])
    if Counter(elements) != Counter(expected) or bonds is None:
        return []
    actual_adjacency = [set() for _ in elements]
    template_adjacency = [set() for _ in expected]
    for i, j in bonds:
        actual_adjacency[int(i)].add(int(j)); actual_adjacency[int(j)].add(int(i))
    for i, j in profile["bonds"]:
        template_adjacency[int(i)].add(int(j)); template_adjacency[int(j)].add(int(i))
    def signature(i, labels, adjacency):
        return labels[i], tuple(sorted(Counter(labels[j] for j in adjacency[i]).items()))
    heavy = [i for i, element in enumerate(expected) if element != "H"]
    candidates = {i: [j for j, element in enumerate(elements) if element != "H"
                      and signature(i, expected, template_adjacency)
                      == signature(j, elements, actual_adjacency)] for i in heavy}
    mapping = dict(zip(profile["nitrile_indices"], map(int, nitrile)))
    if any(j not in candidates[i] for i, j in mapping.items()):
        return []
    results = []
    def compatible(i, j):
        return all((k in template_adjacency[i]) == (v in actual_adjacency[j])
                   for k, v in mapping.items() if k != i)
    if not all(compatible(i, j) for i, j in mapping.items()):
        return []
    def search():
        remaining = [i for i in heavy if i not in mapping]
        if not remaining:
            results.append(dict(mapping)); return
        i = min(remaining, key=lambda k: sum(
            j not in mapping.values() and compatible(k, j) for j in candidates[k]))
        for j in candidates[i]:
            if j not in mapping.values() and compatible(i, j):
                mapping[i] = j; search(); del mapping[i]
    search()
    return results


def _electrostatic_response(r, env, q, e, a, coefficient, roles, nitrile_n, bohr_per_A):
    delta = r[:, None, :]-env[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance <= 0):
        raise ValueError("Coincident solute and environment charge centers")
    q = q/bohr_per_A
    potential = np.sum(q[None, :]/distance, axis=1)
    gradient = -np.einsum("k,ikj,ik->ij", q, delta, distance**-3)
    ep = np.einsum("ij,ikj->ik", e, delta)
    second = np.sum(q[None, :]*(3*ep*ep/distance**5
                    - np.sum(e*e, axis=1)[:, None]/distance**3), axis=1)
    per_atom = coefficient*(second-np.sum(gradient*a, axis=1))
    fixed, electronic = [], []
    for role in range(8):
        chosen = roles == role
        fixed.append(per_atom[chosen].sum()-chosen.sum()/len(r)*per_atom.sum())
        electronic.append(coefficient*np.sum(potential[chosen]-potential[nitrile_n]))
    return np.r_[fixed, electronic]


def _radial_response(r, env, elements, e, a, coefficient, roles, profile):
    d = r[:, None, :]-env[None, :, :]
    distance = np.linalg.norm(d, axis=-1)
    if np.any(distance <= 0):
        raise ValueError("Coincident solute and environment atom centers")
    u = d/distance[:, :, None]
    start, cutoff = profile["switch_start_A"], profile["cutoff_A"]
    width = cutoff-start
    t = np.clip((distance-start)/width, 0, 1)
    switch = 1-10*t**3+15*t**4-6*t**5
    ds = np.where((distance > start)&(distance < cutoff),
                  (-30*t**2+60*t**3-30*t**4)/width, 0.)
    dds = np.where((distance > start)&(distance < cutoff),
                   (-60*t+180*t**2-120*t**3)/width**2, 0.)
    k = profile["exponential_decay_per_A"]
    f = np.exp(-k*distance)
    dv, ddv = f*(ds-k*switch), f*(dds-2*k*ds+k*k*switch)
    eu, au = np.einsum("ic,ijc->ij", e, u), np.einsum("ic,ijc->ij", a, u)
    response = coefficient*((ddv-dv/distance)*eu**2
        + dv/distance*np.sum(e**2, axis=1)[:, None]-dv*au)
    return np.array([response[np.ix_(roles == role, elements == element)].sum()
                     for role in range(8) for element in profile["environment_elements"]])


def correction_features(conformer, environment, charges, elements, profile):

    elements = np.asarray(elements)
    missing = set(elements)-set(profile["environment_elements"])
    if missing:
        raise ValueError(f"BCA correction has no fitted contact basis for elements {sorted(missing)}")
    electrostatic, radial = [], []
    for prefix, n_index in (("native", 7), ("virtual", 5)):
        r, e, a = (conformer[prefix+"_"+name] for name in ("coordinates_A", "e", "a"))
        roles, coefficient = profile[prefix+"_roles"], conformer[prefix+"_Cj"]
        electrostatic.append(_electrostatic_response(
            r, environment, charges, e, a, coefficient, roles, n_index, profile["bohr_per_A"]))
        radial.append(_radial_response(r, environment, elements, e, a, coefficient, roles, profile))
    return np.r_[electrostatic[0]-electrostatic[1], radial[0]-radial[1]]


def prepare_probe(positions, elements, charges, molecule_ids, solute_indices,
                  nitrile_indices, physics, profiles, *, box=None, bonds=None,
                  global_electrostatics=None, response_profile="auto"):
    if response_profile not in {"auto", "acetonitrile", "virtual-acn", "bca"}:
        raise ValueError("response_profile must be auto, acetonitrile, virtual-acn or bca")
    selection = np.asarray(solute_indices)
    if (selection.ndim != 1 or selection.dtype.kind not in "iu" or not len(selection)
            or np.any(selection < 0) or np.any(selection >= len(elements))
            or len(set(selection.tolist())) != len(selection)):
        raise ValueError("Atom indices must be unique, zero-based integers within the frame")
    selected_elements = tuple(canonical_element(elements[i]) for i in solute_indices)
    is_acn = Counter(selected_elements) == Counter({"C": 2, "H": 3, "N": 1})
    if response_profile == "acetonitrile" or (response_profile == "auto" and is_acn):
        return prepare_frame(positions, elements, charges, molecule_ids, solute_indices,
            nitrile_indices, physics, box=box, bonds=bonds, global_electrostatics=global_electrostatics)
    system = _selected_system(positions, elements, charges, molecule_ids, solute_indices,
                             nitrile_indices, box, bonds)
    native, local_elements, local_bonds, nitrile, r, elements, q, molecules, indices, all_bonds, cell = system
    profile = profiles.get("capped-bca")
    mappings = [] if profile is None else _heavy_mappings(local_elements, local_bonds, nitrile, profile)
    if (response_profile == "auto" and profile is not None
            and Counter(local_elements) == Counter(profile["elements"]) and local_bonds is None):
        raise ValueError("Potential capped BCA requires bond connectivity for automatic native-response selection; use response_profile='virtual-acn' for an explicitly unadapted baseline")
    if response_profile == "bca" or (response_profile == "auto" and mappings):
        if not mappings:
            raise ValueError("Native BCA correction requires the studied 23-atom Ac-Bca-NHMe topology and its bonds. Free BCA and peptides can use --probe virtual-acn as an unadapted transfer baseline.")
        if global_electrostatics is not None:
            raise ValueError("External five-component ACN electrostatics do not define the native BCA sixteen-function electrostatic response")
        heavy = np.array([i for i, e in enumerate(profile["elements"]) if e != "H"])
        fits = []
        for name, conformer in profile["conformers"].items():
            for mapping in mappings:
                fit = _rigid_fit(conformer["native_coordinates_A"][heavy],
                                 native[[mapping[int(i)] for i in heavy]])
                fits.append((fit[2], name, fit[0], fit[1], mapping))
        rmsd, name, rotation, translation, mapping = min(fits, key=lambda item: (item[0], item[1]))
        if rmsd > profile["maximum_heavy_atom_rmsd_A"]:
            raise ValueError(f"BCA geometry is {rmsd:.3f} Angstrom RMSD from the nearest supported conformer; its native response is unavailable. Use --probe virtual-acn for an explicitly unadapted baseline.")
        conformer = profile["conformers"][name]
        virtual = conformer["virtual_coordinates_A"] @ rotation+translation
        frame = _baseline(virtual, r, elements, q, molecules, indices, all_bonds, cell,
                          physics, global_electrostatics)


        environment, environment_elements, environment_charges = [], [], []
        midpoint = virtual[[4, 5]].mean(axis=0)
        index_set = set(map(int, indices))
        neighbors = {int(i): set() for i in indices}
        if all_bonds is not None:
            for i, j in all_bonds:
                if int(i) in index_set and int(j) in index_set:
                    neighbors[int(i)].add(int(j)); neighbors[int(j)].add(int(i))
        for molecule in np.unique(molecules[indices]):
            group = indices[molecules[indices] == molecule]
            lookup = {int(i): j for j, i in enumerate(group)}
            local = [(lookup[int(i)], lookup[j]) for i in group
                     for j in neighbors[int(i)] if j in lookup and int(i) < j]
            whole = whole_and_image(r[group], midpoint, cell, local if all_bonds is not None else None)
            environment.extend((whole-translation) @ rotation.T)
            environment_elements.extend(elements[i] for i in group)
            environment_charges.extend(q[group])
        features = correction_features(conformer, np.asarray(environment).reshape(-1, 3),
            np.asarray(environment_charges), environment_elements, profile)
        delta = float(features @ profile["coefficient"])
        return replace(frame, response_profile="capped-bca",
            prediction_scope="retrospective_native_capped_BCA_response_adapter",
            scaffold_correction_cm1=delta,
            transfer_metadata={"conformer": name, "heavy_atom_rmsd_A": rmsd,
                "excluded_solute_atoms": len(native), "native_HF_correction_cm1": delta,
                "operator_application": "nearest_reference_conformer_rigid_projection",
                "new_conformers_or_equilibrium_BCA_spectra_validated": False})
    virtual = _virtual_geometry(native, nitrile, physics)
    frame = _baseline(virtual, r, elements, q, molecules, indices, all_bonds, cell,
                      physics, global_electrostatics)
    return replace(frame, response_profile="virtual-acn",
        prediction_scope="unadapted_ACN_response_at_selected_nitrile_site",
        transfer_metadata={"excluded_solute_atoms": len(native),
            "native_scaffold_response_included": False,
            "native_scaffold_accuracy_validated": False})
