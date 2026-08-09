"""Critical Ising absorbing-diffusion training framework."""

from .model import IsingDiffusionModel
from .objective import AbsorbingDiffusionObjective
from .sampler import sample_absorbing
from .spec import ExperimentSpec, load_experiment
from .system import IsingDiffusionSystem

__all__ = [
    "AbsorbingDiffusionObjective",
    "ExperimentSpec",
    "IsingDiffusionModel",
    "IsingDiffusionSystem",
    "load_experiment",
    "sample_absorbing",
]
__version__ = "0.1.0"
