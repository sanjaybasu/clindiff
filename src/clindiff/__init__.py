"""clindiff: self-hosted statistical change-control verification for clinical AI updates."""
from .stats import (
    betting_evalue_upper,
    betting_upper_bound,
    betting_two_sided_cs,
    gross_downward_flip_stats,
    unbiased_mean_squared_churn,
    minimum_verifiable_margin,
    profile_product_evalue,
    profile_product_upper_bound,
    elond_step,
    ppi_plus_plus_paired,
    judge_calibration_equivalence,
)
from .gates import evaluate_release, hash_manifest, load_profile

__all__ = [
    "betting_evalue_upper",
    "betting_upper_bound",
    "betting_two_sided_cs",
    "gross_downward_flip_stats",
    "unbiased_mean_squared_churn",
    "minimum_verifiable_margin",
    "profile_product_evalue",
    "profile_product_upper_bound",
    "elond_step",
    "ppi_plus_plus_paired",
    "judge_calibration_equivalence",
    "evaluate_release",
    "hash_manifest",
    "load_profile",
]
