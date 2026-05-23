"""Energy Comparison Eval — Compare the energy efficiency of HDC vs SNN vs MLP vs Transformer.

Uses the Horowitz ISSCC 2014 energy model to estimate per-operation
nanojoule costs at 45nm CMOS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evals.base import EvalConfig, EvalResult
from evals.elsuite.prompt.base import SolverEval

# Horowitz 2014 energy per 45nm operation (pJ)
ENERGY_MAP = {
    "add": 0.9,  # pJ for INT32 add
    "mul": 3.7,  # pJ for INT32 multiply
    "mac": 4.6,  # pJ for INT32 multiply-accumulate
    "mem_read": 5.0,  # pJ per SRAM read (64-bit)
    "mem_write": 5.0,  # pJ per SRAM write
}


@dataclass
class EnergyComparisonConfig(EvalConfig):
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    vocab_size: int = 1000
    seq_len: int = 64
    n_heads: int = 4
    hdc_dim: int = 4096
    n_classes: int = 10
    description: str = "Energy efficiency comparison across neural architectures"


class EnergyComparisonEval(SolverEval):
    """Compare energy per inference across architectures."""

    name: str = "energy.comparison"
    config: EnergyComparisonConfig = EnergyComparisonConfig()

    def _load_solver(self):
        return None  # Pure analytical — no solver needed

    def run(self) -> EvalResult:
        cfg = self.config

        # --- HDC energy estimation ---
        hdc_macs = cfg.hdc_dim
        hdc_adds = cfg.hdc_dim + cfg.n_classes * cfg.hdc_dim
        energy_hdc_pj = hdc_macs * ENERGY_MAP["mac"] + hdc_adds * ENERGY_MAP["add"]
        energy_nj_hdc = energy_hdc_pj / 1000.0

        # --- SNN energy estimation (single timestep) ---
        snn_sparsity = 0.1
        snn_macs = cfg.input_dim * cfg.hidden_dim * snn_sparsity
        snn_macs += cfg.hidden_dim * cfg.output_dim * snn_sparsity
        energy_snn_pj = snn_macs * ENERGY_MAP["mac"]
        energy_nj_snn = energy_snn_pj / 1000.0

        # --- MLP energy estimation ---
        mlp_macs = cfg.input_dim * cfg.hidden_dim + cfg.hidden_dim * cfg.output_dim
        energy_mlp_pj = mlp_macs * ENERGY_MAP["mac"]
        energy_nj_mlp = energy_mlp_pj / 1000.0

        # --- Transformer energy estimation ---
        d_model = cfg.hidden_dim
        d_k = d_model // cfg.n_heads
        attn_macs = cfg.seq_len * d_k * cfg.seq_len * cfg.n_heads
        attn_macs += cfg.seq_len * cfg.seq_len * d_model
        ffn_macs = 2 * d_model * 4 * d_model
        embed_macs = cfg.vocab_size * d_model + d_model * cfg.output_dim
        total_macs = attn_macs + ffn_macs + embed_macs
        energy_transformer_pj = total_macs * ENERGY_MAP["mac"]
        energy_nj_transformer = energy_transformer_pj / 1000.0
        energy_uj_transformer = energy_nj_transformer / 1000.0

        return EvalResult(
            name=self.name,
            metrics={
                "energy_nJ_hdc": energy_nj_hdc,
                "energy_nJ_snn": energy_nj_snn,
                "energy_nJ_mlp": energy_nj_mlp,
                "energy_nJ_transformer": energy_nj_transformer,
                "energy_uJ_transformer": energy_uj_transformer,
                "ratio_hdc_vs_transformer": (
                    energy_nj_transformer / energy_nj_hdc if energy_nj_hdc > 0 else 0
                ),
                "ratio_snn_vs_transformer": (
                    energy_nj_transformer / energy_nj_snn if energy_nj_snn > 0 else 0
                ),
                "ratio_hdc_vs_mlp": (
                    energy_nj_mlp / energy_nj_hdc if energy_nj_hdc > 0 else 0
                ),
                "ratio_snn_vs_mlp": (
                    energy_nj_mlp / energy_nj_snn if energy_nj_snn > 0 else 0
                ),
            },
            metadata={"config": cfg.to_dict()},
        )
