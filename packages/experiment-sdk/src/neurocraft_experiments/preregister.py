"""Matched, deterministic trial matrices fixed before behavioral execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .identity import RunMode, canonical_sha256


@dataclass(frozen=True, slots=True)
class TrialCell:
    arena_id: str
    mode: RunMode
    seed: int
    replicate: int
    intervention: str = "none"

    def __post_init__(self) -> None:
        if not self.arena_id or self.seed < 0 or self.replicate < 0:
            raise ValueError("trial cell identifiers must be non-empty and non-negative")
        if self.mode is RunMode.LESION and self.intervention == "none":
            raise ValueError("lesion cells require a named intervention")
        if self.mode is not RunMode.LESION and self.intervention != "none":
            raise ValueError("only lesion cells may carry an intervention")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    plan_id: str
    hypothesis: str
    primary_metric: str
    cells: tuple[TrialCell, ...]

    @classmethod
    def matched(
        cls,
        *,
        plan_id: str,
        hypothesis: str,
        primary_metric: str,
        arenas: Iterable[str],
        modes: Iterable[RunMode],
        seeds: Iterable[int],
        replicates: int,
    ) -> ExperimentPlan:
        if replicates <= 0:
            raise ValueError("replicates must be positive")
        ordered_arenas = tuple(sorted(set(arenas)))
        ordered_modes = tuple(sorted(set(modes), key=lambda mode: mode.value))
        ordered_seeds = tuple(sorted(set(seeds)))
        if not ordered_arenas or not ordered_modes or not ordered_seeds:
            raise ValueError("matched plans require arenas, modes, and seeds")
        cells = tuple(
            TrialCell(arena, mode, seed, replicate)
            for arena in ordered_arenas
            for seed in ordered_seeds
            for replicate in range(replicates)
            for mode in ordered_modes
        )
        return cls(plan_id, hypothesis, primary_metric, cells)

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "plan_id": self.plan_id,
                "hypothesis": self.hypothesis,
                "primary_metric": self.primary_metric,
                "cells": [
                    {
                        "arena_id": cell.arena_id,
                        "mode": cell.mode.value,
                        "seed": cell.seed,
                        "replicate": cell.replicate,
                        "intervention": cell.intervention,
                    }
                    for cell in self.cells
                ],
            }
        )
