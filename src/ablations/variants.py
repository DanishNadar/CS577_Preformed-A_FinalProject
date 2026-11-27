# Loss-ablation variants and the per-run config overrides that keep every comparison fair

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import ExperimentConfig, configToDict, loadExperimentConfig

DEFAULT_SEED = 577
DEFAULT_OUTPUT_ROOT = "results/ablations"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/ablations"
DEFAULT_DATA_ROOT = "data/synthetic"

# Config sections that may differ between runs of one comparison; everything else must match exactly
RUN_SPECIFIC_KEYS = ("experimentName", "outputDir", "checkpointDir", "loss")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    lambdaCoord: float
    lambdaNoobj: float
    useSqrtWidthHeight: bool
    description: str

    def lossOverrides(self) -> dict:
        return {
            "loss.lambdaCoord": self.lambdaCoord,
            "loss.lambdaNoobj": self.lambdaNoobj,
            "loss.useSqrtWidthHeight": self.useSqrtWidthHeight,
        }


VARIANTS: dict[str, AblationVariant] = {
    variant.name: variant
    for variant in (
        AblationVariant("paper", 5.0, 0.5, True, "Paper loss: lambdaCoord=5, lambdaNoobj=0.5, sqrt(w), sqrt(h)"),
        AblationVariant("unitWeights", 1.0, 1.0, True, "Ablation 1 (proposal): both weights set to 1"),
        AblationVariant("directWh", 5.0, 0.5, False, "Ablation 2 (proposal): regress w, h directly, no sqrt"),
        AblationVariant("unitCoord", 1.0, 0.5, True, "Factorial: only lambdaCoord set to 1"),
        AblationVariant("unitNoobj", 5.0, 1.0, True, "Factorial: only lambdaNoobj set to 1"),
        AblationVariant("unitWeightsDirectWh", 1.0, 1.0, False, "Factorial: both ablations together"),
    )
}
PROPOSAL_VARIANTS = ("paper", "unitWeights", "directWh")
REFERENCE_VARIANT = "paper"


def dataRootForSeed(seed: int) -> str:
    # The seed fixes the data split too, so each seed gets its own dataset folder; 577 reuses the baseline's
    return DEFAULT_DATA_ROOT if seed == DEFAULT_SEED else f"{DEFAULT_DATA_ROOT}_seed{seed}"


def runDirectoryName(variantName: str, seed: int) -> str:
    return f"seed{seed}/{variantName}"


def variantOverrides(
    variantName: str,
    seed: int = DEFAULT_SEED,
    outputRoot: str | Path = DEFAULT_OUTPUT_ROOT,
    checkpointRoot: str | Path = DEFAULT_CHECKPOINT_ROOT,
    extraOverrides: dict | None = None,
) -> dict:
    # Dotted overrides for one (variant, seed) run; extraOverrides apply to all runs and must not touch loss.*
    variant = VARIANTS[variantName]
    extraOverrides = dict(extraOverrides or {})
    lossKeys = [key for key in extraOverrides if key.startswith("loss.")]
    if lossKeys:
        raise ValueError(f"Shared overrides must not change the loss (would break the ablation): {lossKeys}")
    runName = runDirectoryName(variantName, seed)
    overrides = {
        "data.synthetic.root": dataRootForSeed(seed),
        **extraOverrides,
        "experimentName": f"ablation-{variantName}-seed{seed}",
        "seed": seed,
        "outputDir": str(Path(outputRoot) / runName),
        "checkpointDir": str(Path(checkpointRoot) / runName),
        **variant.lossOverrides(),
    }
    return overrides


def buildVariantConfig(
    variantName: str,
    seed: int = DEFAULT_SEED,
    baseConfigPath: str | Path = "configs/baseline.yaml",
    outputRoot: str | Path = DEFAULT_OUTPUT_ROOT,
    checkpointRoot: str | Path = DEFAULT_CHECKPOINT_ROOT,
    extraOverrides: dict | None = None,
) -> ExperimentConfig:
    overrides = variantOverrides(variantName, seed, outputRoot, checkpointRoot, extraOverrides)
    return loadExperimentConfig(baseConfigPath, overrides)


def sharedSettings(config: ExperimentConfig) -> dict:
    # Everything that must be identical between variants trained with the same seed
    values = configToDict(config)
    return {key: value for key, value in values.items() if key not in RUN_SPECIFIC_KEYS}


def assertFairComparison(configs: list[ExperimentConfig]) -> None:
    # Runs that share a seed may differ only in the loss and their output folders
    bySeed: dict[int, list[ExperimentConfig]] = {}
    for config in configs:
        bySeed.setdefault(config.seed, []).append(config)
    for seed, seedConfigs in bySeed.items():
        reference = sharedSettings(seedConfigs[0])
        for config in seedConfigs[1:]:
            settings = sharedSettings(config)
            if settings != reference:
                differing = sorted(key for key in reference if reference[key] != settings.get(key))
                raise ValueError(f"Seed {seed}: '{config.experimentName}' differs from "
                                 f"'{seedConfigs[0].experimentName}' outside the loss in {differing}")
