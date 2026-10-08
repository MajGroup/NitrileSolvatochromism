from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ELEMENTS = ("PAD", "H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "OTHER")

NONPADDING_ELEMENTS = ELEMENTS[1:]

ELEMENT_ENCODING = "one_hot_nonpadding_canonical_element_index_v1"

CONTINUOUS_FEATURES = ("q_zero", "r_C_A", "r_N_A", "z_A", "rho_A")

GLOBAL_ELECTROSTATIC_FEATURES = (
    "el_obw_dV_C_minus_N_q_per_A",
    "el_obw_dV_lone_pair_minus_N_q_per_A",
    "el_obw_dV_ring_minus_N_q_per_A",
    "el_field_N_parallel_q_per_A2",
    "el_field_gradient_parallel_q_per_A3",
)

@runtime_checkable
class PreparedFrame(Protocol):


    global_x: np.ndarray
    element_index: np.ndarray
    continuous: np.ndarray
    atom_mask: np.ndarray
    aligned_combined_coordinates_A: np.ndarray
    environment_elements: tuple[str, ...]
    molecule_index: np.ndarray
    whole_tt6_response: float
    per_molecule_tt6_response: np.ndarray

@dataclass(frozen=True)
class SwitchSpec:
    onset_A: float
    cutoff_A: float
    distance_definition: str = "minimum_atom_to_nitrile_C_or_N_A"

    def validate(self) -> None:
        if not np.isfinite(self.onset_A) or not np.isfinite(self.cutoff_A):
            raise ValueError("switch bounds must be finite")
        if self.onset_A < 0.0 or self.cutoff_A <= self.onset_A:
            raise ValueError("invalid switch bounds")
        if self.distance_definition != "minimum_atom_to_nitrile_C_or_N_A":
            raise ValueError("Unsupported electrostatic switch distance definition")

@dataclass(frozen=True)
class MolecularSpec:
    width: int = 96
    electrostatic_local_cutoff_A: float = 7.0
    exchange_cutoff_A: float = 7.0
    delocalization_cutoff_A: float = 7.0
    dispersion_local_cutoff_A: float = 8.0
    cutoff_width_A: float = 1.0
    use_explicit_dispersion: bool = True
    sign_constrained: bool = True
    element_encoding: str = ELEMENT_ENCODING

    @property
    def cutoffs(self) -> tuple[float, float, float, float]:
        return (
            self.electrostatic_local_cutoff_A,
            self.exchange_cutoff_A,
            self.delocalization_cutoff_A,
            self.dispersion_local_cutoff_A,
        )

    @property
    def atom_input_dim(self) -> int:
        return len(NONPADDING_ELEMENTS) + len(CONTINUOUS_FEATURES)

    def validate(self) -> None:
        if self.width != 96 or self.element_encoding != ELEMENT_ENCODING:
            raise ValueError("Model architecture does not match the saved parameters")
        if self.cutoff_width_A <= 0.0 or any(
            cutoff <= self.cutoff_width_A for cutoff in self.cutoffs
        ):
            raise ValueError("invalid molecular switch")

@dataclass(frozen=True)
class ElectrostaticSpec:
    width: int = 96
    switch: SwitchSpec = SwitchSpec(6.0, 7.0)
    experiment_type: str = "controlled_localq0_onehot_element_electrostatic_residual_v1"
    element_encoding: str = ELEMENT_ENCODING
    element_vocabulary: tuple[str, ...] = NONPADDING_ELEMENTS
    continuous_features: tuple[str, ...] = CONTINUOUS_FEATURES

    def validate(self) -> None:
        self.switch.validate()
        if (
            self.width != 96
            or self.experiment_type
            != "controlled_localq0_onehot_element_electrostatic_residual_v1"
            or self.element_encoding != ELEMENT_ENCODING
            or tuple(self.element_vocabulary) != NONPADDING_ELEMENTS
            or tuple(self.continuous_features) != CONTINUOUS_FEATURES
        ):
            raise ValueError("Electrostatic model settings do not match the saved parameters")

    @property
    def atom_input_dim(self) -> int:
        return len(self.element_vocabulary) + len(self.continuous_features)

class _VocabularyMetadata(nn.Module):


    def __init__(self, n_elements: int) -> None:
        super().__init__()
        self.num_embeddings = int(n_elements)

def one_hot_nonpadding(element: torch.Tensor) -> torch.Tensor:
    if element.dtype != torch.long:
        element = element.long()
    if torch.any(element.lt(0)) or torch.any(element.ge(len(ELEMENTS))):
        raise ValueError("Element index is outside the supported range")
    return F.one_hot(element, num_classes=len(ELEMENTS))[..., 1:].to(torch.float32)

def cosine_component_gate(
    distance_A: torch.Tensor, cutoffs_A: torch.Tensor, width_A: float
) -> torch.Tensor:
    onset = cutoffs_A - width_A
    x = ((distance_A.unsqueeze(-1) - onset) / max(width_A, 1.0e-8)).clamp(0.0, 1.0)
    smooth = 0.5 * (1.0 + torch.cos(math.pi * x))
    return torch.where(
        distance_A.unsqueeze(-1) <= onset,
        torch.ones_like(smooth),
        torch.where(
            distance_A.unsqueeze(-1) < cutoffs_A,
            smooth,
            torch.zeros_like(smooth),
        ),
    )

def complementary_cosine_switch(
    distance_A: torch.Tensor, onset_A: float, cutoff_A: float
) -> torch.Tensor:
    if cutoff_A <= onset_A:
        raise ValueError("cutoff_A must exceed onset_A")
    x = ((distance_A - onset_A) / (cutoff_A - onset_A)).clamp(0.0, 1.0)
    transition = 0.5 * (1.0 + torch.cos(math.pi * x))
    return torch.where(
        distance_A <= onset_A,
        torch.ones_like(distance_A),
        torch.where(distance_A < cutoff_A, transition, torch.zeros_like(distance_A)),
    )

def minimum_endpoint_distance(
    continuous: torch.Tensor, atom_mask: torch.Tensor
) -> torch.Tensor:
    if continuous.ndim != 3 or continuous.shape[-1] != 5:
        raise ValueError("continuous must have shape (molecule, atom, 5)")
    if atom_mask.shape != continuous.shape[:2]:
        raise ValueError("Atom mask and descriptor shapes do not match")
    endpoint = torch.minimum(continuous[..., 1], continuous[..., 2])
    endpoint = endpoint.masked_fill(~atom_mask.bool(), float("inf"))
    distance = endpoint.min(dim=1).values
    if not torch.isfinite(distance).all():
        raise ValueError("a molecule contains no occupied atom")
    return distance

class MolecularNet(nn.Module):


    def __init__(
        self,
        spec: MolecularSpec,
        n_elements: int,
        global_scale: np.ndarray,
        dispersion_scale: float,
        target_scale: np.ndarray,
    ) -> None:
        super().__init__()
        spec.validate()
        if int(n_elements) != len(ELEMENTS):
            raise ValueError("Unexpected number of supported elements")
        self.spec = spec
        self.n_elements = int(n_elements)
        self.embedding = _VocabularyMetadata(self.n_elements)
        self.atom_encoder = nn.Sequential(
            nn.Linear(spec.atom_input_dim, spec.width),
            nn.SiLU(),
            nn.LayerNorm(spec.width),
            nn.Linear(spec.width, spec.width),
            nn.SiLU(),
        )
        self.molecule_head = nn.Sequential(
            nn.Linear(3 * spec.width, spec.width),
            nn.SiLU(),
            nn.LayerNorm(spec.width),
            nn.Linear(spec.width, spec.width),
            nn.SiLU(),
            nn.Linear(spec.width, 4),
        )
        self.global_electrostatic = nn.Linear(
            len(GLOBAL_ELECTROSTATIC_FEATURES), 1, bias=False
        )
        self.global_dispersion = (
            nn.Linear(1, 1, bias=False) if spec.use_explicit_dispersion else None
        )
        self.register_buffer(
            "global_scale", torch.as_tensor(global_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "dispersion_scale",
            torch.as_tensor([dispersion_scale], dtype=torch.float32),
        )
        self.register_buffer(
            "target_scale", torch.as_tensor(target_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "cutoffs", torch.as_tensor(spec.cutoffs, dtype=torch.float32)
        )

    def atom_features(
        self, element: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        if continuous.shape[:-1] != element.shape or continuous.shape[-1] != 5:
            raise ValueError("Descriptor and element-array shapes do not match")
        normalized = torch.stack(
            (
                continuous[..., 0].clamp(-2.0, 2.0),
                continuous[..., 1] / 8.0,
                continuous[..., 2] / 8.0,
                continuous[..., 3] / 8.0,
                continuous[..., 4] / 8.0,
            ),
            dim=-1,
        )
        identity = one_hot_nonpadding(element).to(
            dtype=normalized.dtype, device=normalized.device
        )
        return torch.cat((identity, normalized), dim=-1)

    def encoded_pool(
        self,
        element: torch.Tensor,
        continuous: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.atom_encoder(self.atom_features(element, continuous))
        atom_weight = atom_mask.unsqueeze(-1)
        count = atom_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        summed = (hidden * atom_weight).sum(dim=2)
        mean = summed / count
        root_sum = summed / torch.sqrt(count)
        maximum = hidden.masked_fill(~atom_mask.bool().unsqueeze(-1), -1.0e9).max(
            dim=2
        ).values
        present = atom_mask.max(dim=2).values.unsqueeze(-1)
        maximum = torch.where(present.bool(), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, root_sum, maximum), dim=-1), present

    def molecule_components(
        self,
        element: torch.Tensor,
        continuous: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        latent, present = self.encoded_pool(element, continuous, atom_mask)
        raw = self.molecule_head(latent)
        if self.spec.sign_constrained:
            raw = torch.stack(
                (
                    raw[..., 0],
                    F.softplus(raw[..., 1]),
                    -F.softplus(raw[..., 2]),
                    -F.softplus(raw[..., 3]),
                ),
                dim=-1,
            )
        radial = torch.sqrt(
            torch.clamp(
                continuous[..., 3].square() + continuous[..., 4].square(), min=0.0
            )
        ).masked_fill(~atom_mask.bool(), 1.0e6)
        nearest = radial.min(dim=2).values
        gate = cosine_component_gate(
            nearest, self.cutoffs, self.spec.cutoff_width_A
        )
        return raw * self.target_scale * gate * present

    def shell_prediction(
        self,
        molecule_components: torch.Tensor,
        molecule_mask: torch.Tensor,
        global_x: torch.Tensor,
        dispersion_x: torch.Tensor,
    ) -> torch.Tensor:
        output = (molecule_components * molecule_mask.unsqueeze(-1)).sum(dim=1)
        output = output.clone()
        output[:, 0:1] = (
            output[:, 0:1]
            + self.global_electrostatic(global_x / self.global_scale)
            * self.target_scale[0]
        )
        if self.global_dispersion is not None:
            output[:, 3:4] = (
                output[:, 3:4]
                + self.global_dispersion(dispersion_x / self.dispersion_scale)
                * self.target_scale[3]
            )
        return output

class ElectrostaticNet(nn.Module):


    def __init__(self, spec: ElectrostaticSpec, residual_scale_cm1: float) -> None:
        super().__init__()
        spec.validate()
        scale = float(residual_scale_cm1)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("residual scale must be finite and positive")
        self.spec = spec
        self.register_buffer(
            "residual_scale_cm1", torch.tensor(scale, dtype=torch.float32)
        )
        self.atom_encoder = nn.Sequential(
            nn.Linear(spec.atom_input_dim, spec.width),
            nn.SiLU(),
            nn.LayerNorm(spec.width),
            nn.Linear(spec.width, spec.width),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(3 * spec.width, spec.width),
            nn.SiLU(),
            nn.LayerNorm(spec.width),
            nn.Linear(spec.width, spec.width),
            nn.SiLU(),
            nn.Linear(spec.width, 1),
        )

    def raw_residual(
        self,
        element: torch.Tensor,
        continuous: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if element.shape != atom_mask.shape or continuous.shape[:2] != atom_mask.shape:
            raise ValueError("Electrostatic input-array shapes do not match")
        if continuous.shape[-1] != 5:
            raise ValueError("continuous tensor must end in five channels")
        occupied = atom_mask.bool()
        if torch.any(occupied & element.eq(0)):
            raise ValueError("occupied atoms cannot use PAD")
        if torch.any(element.lt(0)) or torch.any(element.ge(len(ELEMENTS))):
            raise ValueError("Element index is outside the supported range")
        if torch.any(continuous[..., 0].masked_select(occupied).ne(0.0)):
            raise ValueError("the frozen one-hot model requires local q=0")
        distance = minimum_endpoint_distance(continuous, atom_mask)
        normalized = torch.stack(
            (
                continuous[..., 0],
                continuous[..., 1] / 8.0,
                continuous[..., 2] / 8.0,
                continuous[..., 3] / 8.0,
                continuous[..., 4] / 8.0,
            ),
            dim=-1,
        )
        identity = one_hot_nonpadding(element).to(
            dtype=normalized.dtype, device=normalized.device
        )
        encoded = self.atom_encoder(torch.cat((identity, normalized), dim=-1))
        weight = occupied.to(encoded.dtype).unsqueeze(-1)
        count = occupied.to(encoded.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        summed = (encoded * weight).sum(dim=1)
        mean = summed / count
        root_sum = summed / torch.sqrt(count)
        maximum = encoded.masked_fill(~occupied.unsqueeze(-1), -1.0e9).max(
            dim=1
        ).values
        raw = self.residual_head(torch.cat((mean, root_sum, maximum), dim=-1))[:, 0]
        return raw * self.residual_scale_cm1, distance

@dataclass(frozen=True)
class SpinHead:
    coefficient_physical: np.ndarray
    cutoff_A: float
    cutoff_width_A: float
    os_fraction: float

@dataclass(frozen=True)
class Ensemble:
    hf_models: tuple[MolecularNet, ...]
    spin_heads: tuple[SpinHead, ...]
    electro_models: tuple[ElectrostaticNet, ...]
    electro_coefficients: np.ndarray

def cosine_spin_gate(distance: np.ndarray, cutoff_A: float, width_A: float) -> np.ndarray:
    onset = cutoff_A - width_A
    values = np.asarray(distance, dtype=np.float64)
    x = np.clip((values - onset) / width_A, 0.0, 1.0)
    smooth = 0.5 * (1.0 + np.cos(np.pi * x))
    return np.where(values <= onset, 1.0, np.where(values < cutoff_A, smooth, 0.0))

def heavy_atom_midpoint_distances(frame: PreparedFrame) -> np.ndarray:
    coordinates = np.asarray(frame.aligned_combined_coordinates_A[6:], dtype=np.float64)
    molecule_index = np.asarray(frame.molecule_index, dtype=np.int64)
    if coordinates.shape != (len(frame.environment_elements), 3):
        raise RuntimeError("Environment coordinate and element counts do not match")
    if molecule_index.shape != (len(frame.environment_elements),):
        raise RuntimeError("Molecule-index and element counts do not match")
    heavy = np.asarray(
        [str(element) not in ("H", "D") for element in frame.environment_elements],
        dtype=bool,
    )
    n_molecules = len(frame.element_index)
    result = np.full(n_molecules, np.inf, dtype=np.float64)
    radial = np.linalg.norm(coordinates, axis=1)
    for molecule in range(n_molecules):
        selected = (molecule_index == molecule) & heavy
        if not np.any(selected):
            raise RuntimeError(f"environment molecule {molecule} has no heavy atom")
        result[molecule] = float(np.min(radial[selected]))
    if not np.isfinite(result).all():
        raise RuntimeError("heavy-atom midpoint distances are non-finite")
    return result


def _validate_frame(frame: PreparedFrame) -> tuple[np.ndarray, np.ndarray]:
    element = np.asarray(frame.element_index)
    continuous = np.asarray(frame.continuous)
    raw_mask = np.asarray(frame.atom_mask)
    if element.dtype.kind not in "iu":
        raise RuntimeError("prepared element indices must have integer dtype")
    if raw_mask.dtype.kind not in "biuf" or not np.isfinite(raw_mask).all():
        raise RuntimeError("prepared atom mask must be finite and binary")
    if not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise RuntimeError("prepared atom mask contains a non-binary value")
    mask = raw_mask.astype(bool, copy=False)
    if element.ndim != 2 or mask.shape != element.shape or continuous.shape != (*element.shape, 5):
        raise RuntimeError("Prepared coordinate, element and mask shapes do not match")
    if len(element) == 0 or np.any(mask.sum(axis=1) == 0):
        raise RuntimeError("prepared frame contains an empty molecule")
    if (
        not np.isfinite(continuous).all()
        or np.any(element < 0)
        or np.any(element >= len(ELEMENTS))
        or np.any(mask & (element == 0))
        or np.any((~mask) & (element != 0))
    ):
        raise RuntimeError("Prepared frame contains invalid element indices or descriptors")


    if np.any(np.diff(mask.astype(np.int8), axis=1) > 0):
        raise RuntimeError("prepared occupied atom slots are not contiguous")
    molecule_index = np.asarray(frame.molecule_index)
    if molecule_index.dtype.kind not in "iu":
        raise RuntimeError("prepared molecule indices must have integer dtype")
    if (
        molecule_index.shape != (len(frame.environment_elements),)
        or np.any(molecule_index < 0)
        or np.any(molecule_index >= len(element))
        or set(molecule_index.tolist()) != set(range(len(element)))
    ):
        raise RuntimeError("Environment molecule indices are incomplete or out of range")
    global_x = np.asarray(frame.global_x, dtype=np.float64)
    per_molecule_tt6 = np.asarray(frame.per_molecule_tt6_response, dtype=np.float64)
    if (
        global_x.shape != (5,)
        or per_molecule_tt6.shape != (len(element),)
        or not np.isfinite(global_x).all()
        or not np.isfinite(per_molecule_tt6).all()
        or not np.isfinite(float(frame.whole_tt6_response))
    ):
        raise RuntimeError("Electrostatic or dispersion features have invalid shapes or values")
    return mask, global_x

def predict_frame(
    arm: Ensemble, frame: PreparedFrame, device: torch.device | str = "cpu"
) -> tuple:


    torch_device = torch.device(device)
    occupied, global_x_np = _validate_frame(frame)
    continuous = np.array(frame.continuous, dtype=np.float32, copy=True)
    continuous[..., 0][occupied] = 0.0

    element = torch.tensor(frame.element_index, dtype=torch.long, device=torch_device)
    continuous_t = torch.tensor(continuous, dtype=torch.float32, device=torch_device)
    atom_mask = torch.tensor(frame.atom_mask, dtype=torch.float32, device=torch_device)
    fixed_g = float(np.dot(global_x_np, arm.electro_coefficients))
    electro_members: list[float] = []
    with torch.inference_mode():
        for model in arm.electro_models:
            raw, distance = model.raw_residual(element, continuous_t, atom_mask)
            gate = complementary_cosine_switch(
                distance, model.spec.switch.onset_A, model.spec.switch.cutoff_A
            )
            electro_members.append(
                fixed_g
                + float(
                    np.sum(
                        (gate * raw).detach().cpu().numpy().astype(np.float64),
                        dtype=np.float64,
                    )
                )
            )
    corrected_electrostatic = float(np.mean(electro_members, dtype=np.float64))

    element_b = element.unsqueeze(0)
    continuous_b = continuous_t.unsqueeze(0)
    atom_mask_b = atom_mask.unsqueeze(0)
    molecule_mask = torch.ones(
        (1, len(frame.element_index)), dtype=torch.float32, device=torch_device
    )
    global_x = torch.tensor(global_x_np[None, :], dtype=torch.float32, device=torch_device)
    dispersion_x = torch.tensor(
        [[frame.whole_tt6_response]], dtype=torch.float32, device=torch_device
    )
    midpoint_distance = heavy_atom_midpoint_distances(frame)
    base_rows: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    heads: list[np.ndarray] = []
    with torch.inference_mode():
        for model, adapter in zip(arm.hf_models, arm.spin_heads, strict=True):
            local = model.molecule_components(element_b, continuous_b, atom_mask_b)
            base = (
                model.shell_prediction(local, molecule_mask, global_x, dispersion_x)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            latent, _present = model.encoded_pool(element_b, continuous_b, atom_mask_b)
            latent_np = latent[0].detach().cpu().numpy().astype(np.float64)
            if latent_np.shape[1] + 1 != adapter.coefficient_physical.shape[1]:
                raise RuntimeError("HF representation width does not match the correlation coefficients")
            design = np.concatenate(
                (
                    np.asarray(frame.per_molecule_tt6_response, dtype=np.float64)[:, None],
                    latent_np,
                ),
                axis=1,
            )
            spin_gate = cosine_spin_gate(
                midpoint_distance, adapter.cutoff_A, adapter.cutoff_width_A
            )
            correction = np.sum(
                (design @ adapter.coefficient_physical.T) * spin_gate[:, None],
                axis=0,
                dtype=np.float64,
            )
            current_heads = np.asarray(
                (
                    corrected_electrostatic + base[1] + base[2] + correction[0],
                    adapter.os_fraction * base[3] + correction[1],
                    (1.0 - adapter.os_fraction) * base[3] + correction[2],
                ),
                dtype=np.float64,
            )
            base_rows.append(base)
            corrections.append(correction)
            heads.append(current_heads)
    return (corrected_electrostatic, np.stack(base_rows),
            np.stack(corrections), np.stack(heads))
