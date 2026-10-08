import warnings

import numpy as np
import torch

from . import network
from .transfer import prepare_probe


def arrays(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, dict):
        return {key: arrays(item) for key, item in value.items()}
    return value


class NitrileModel:
    def __init__(self, filename, device="cpu"):
        data = torch.load(filename, map_location="cpu", weights_only=True)
        self.device = device
        self.physics = arrays(data["physics"])
        self.profiles = arrays(data.get("response_profiles", {}))
        hf_models, electro_models, spin_heads = [], [], []

        for item in data["hf_members"]:
            state = item["state_dict"]
            model = network.MolecularNet(
                network.MolecularSpec(**item["spec"]), len(network.ELEMENTS),
                state["global_scale"].numpy(), state["dispersion_scale"].item(),
                state["target_scale"].numpy(),
            )
            model.load_state_dict(state)
            hf_models.append(model.requires_grad_(False).eval().to(device))

        for item in data["electrostatic_members"]:
            spec = dict(item["spec"])
            spec["switch"] = network.SwitchSpec(**spec["switch"])
            model = network.ElectrostaticNet(
                network.ElectrostaticSpec(**spec),
                item["state_dict"]["residual_scale_cm1"].item(),
            )
            model.load_state_dict(item["state_dict"])
            electro_models.append(model.requires_grad_(False).eval().to(device))

        for item in data["spin_members"]:
            spin_heads.append(network.SpinHead(
                item["coefficient_physical"].numpy(), item["cutoff_A"],
                item["cutoff_width_A"], item["os_fraction"],
            ))
        if not len(hf_models) == len(electro_models) == len(spin_heads) == 3:
            raise ValueError("The model needs three ensemble members")
        self.ensemble = network.Ensemble(
            tuple(hf_models), tuple(spin_heads), tuple(electro_models),
            data["electrostatic_coefficients"].numpy(),
        )

    @classmethod
    def from_file(cls, filename, device="cpu"):
        return cls(filename, device)

    def prepare(self, positions, elements, charges, molecule_ids, solute_indices,
                nitrile_indices, *, box=None, bonds=None, response_profile="auto",
                global_electrostatics=None):
        return prepare_probe(
            positions, elements, charges, molecule_ids, solute_indices,
            nitrile_indices, self.physics, self.profiles, box=box, bonds=bonds,
            response_profile=response_profile, global_electrostatics=global_electrostatics,
        )

    def predict_prepared(self, frame, return_components=True):
        if len(frame.element_index):
            el, base, correction, heads = network.predict_frame(self.ensemble, frame, self.device)
            totals = heads[:, 0] + 1.2 * heads[:, 1] + (1.0 / 3.0) * heads[:, 2]
            # Keep the components from the member with the median SCS-MP2 total.
            member = int(np.argsort(totals, kind="stable")[1])
            exchange = float(base[member, 1])
            relaxation = float(base[member, 2] + correction[member, 0])
            dispersion = float(1.2 * heads[member, 1] + (1.0 / 3.0) * heads[member, 2])
            source_total = float(totals[member])
        else:
            el = float(frame.global_x @ self.ensemble.electro_coefficients)
            exchange, relaxation, dispersion = 0.0, 0.0, 0.0
            source_total = el

        delta = float(frame.scaffold_correction_cm1)
        result = {
            "electrostatic_cm1": el,
            "exchange_repulsion_cm1": exchange,
            "orbital_relaxation_cm1": relaxation,
            "dispersion_correlation_cm1": dispersion,
            "scaffold_correction_cm1": delta,
            "shift_cm1": source_total + delta,
            "response_profile": frame.response_profile,
            "conformer": (frame.transfer_metadata or {}).get("conformer", ""),
        }
        return result if return_components else result["shift_cm1"]

    def predict(self, positions, elements, charges, molecule_ids, solute_indices,
                nitrile_indices, *, box=None, bonds=None, response_profile="auto",
                return_components=True, global_electrostatics=None):
        frame = self.prepare(
            positions, elements, charges, molecule_ids, solute_indices, nitrile_indices,
            box=box, bonds=bonds, response_profile=response_profile,
            global_electrostatics=global_electrostatics,
        )
        if frame.response_profile == "virtual-acn" and response_profile == "auto":
            warnings.warn("Using the ACN map for this molecule; no native correction is available.")
        return self.predict_prepared(frame, return_components=return_components)
