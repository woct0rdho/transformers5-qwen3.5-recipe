"""Canonical fitted expert-route laws and deterministic route-vector banks."""

import hashlib
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

REFERENCE_TOKENS = 2048
EXPERTS = 256
BASE_SEED = 8_314_159
TOKEN_SEED_STRIDE = 104_729
HASH_SEED_OFFSET = 31_000


class ExpertPrior(str, Enum):
    QwenLearned = "qwen-learned"
    Qwen38Learned = "qwen3.8-learned"
    DeepSeekLearned = "deepseek-learned"
    DeepSeekHash = "deepseek-hash"


EXPERT_PRIOR_NAMES = tuple(item.value for item in ExpertPrior)


@dataclass(frozen=True)
class Residual:
    degrees: float
    location: float
    scale: float
    lower: float
    upper: float

    def sample(self, rng: np.random.Generator) -> float:
        value = self.location + self.scale * float(rng.standard_t(self.degrees))
        return min(self.upper, max(self.lower, value))


@dataclass(frozen=True)
class LearnedLaw:
    top_k: int
    shift: float
    head_multiplier: float
    active: tuple[float, float]
    alpha: tuple[float, float, float]
    active_residual: Residual
    alpha_residual: Residual
    experts: int = EXPERTS


QWEN_LEARNED = LearnedLaw(
    8,
    11.5465232873,
    1.1255020052,
    (1.3568429238, 1.1367804236),
    (0.7376182182, -0.0215993943, -0.1730074363),
    Residual(8.3978226526, -0.0335118652, 0.9830493404, -2.5036492445, 3.7809143778),
    Residual(5.2038953632, 0.0054814884, 0.1025706842, -0.4463245132, 0.3833201702),
)
DEEPSEEK_LEARNED = LearnedLaw(
    6,
    6.3223820835,
    1.0643729189,
    (2.2468973539, 0.8906164814),
    (0.4081577973, -0.0211298377, -0.0754167931),
    Residual(33.5987235964, -0.0013194870, 0.8328268568, -1.6829685257, 2.7587218851),
    Residual(5.1551932778, -0.0071110386, 0.0923527684, -0.3180690433, 0.3604795507),
)
QWEN38_LEARNED = LearnedLaw(
    10,
    20.7163112483,
    1.1946652586,
    (0.7076361999, 0.9702739281),
    (0.7505135414, -0.0229817472, -0.2302367069),
    Residual(7.4647474580, 0.0130546014, 0.4915605168, -1.4935692978, 1.9212931553),
    Residual(8469463127.5627, 0.0002737558, 0.0870558989, -0.2081819082, 0.2166008918),
    experts=512,
)
LEARNED_LAWS = {
    ExpertPrior.QwenLearned: QWEN_LEARNED,
    ExpertPrior.Qwen38Learned: QWEN38_LEARNED,
    ExpertPrior.DeepSeekLearned: DEEPSEEK_LEARNED,
}
LEARNED_FAMILIES = {
    ExpertPrior.QwenLearned: "qwen",
    ExpertPrior.Qwen38Learned: "qwen3.8",
    ExpertPrior.DeepSeekLearned: "deepseek",
}
HASH_COEFFICIENTS = {
    "mu_rho": 0.076568603515625,
    "kappa_rho": 111.40091020461985,
    "a0": 6.660030508273053,
    "beta": 0.4771750368253468,
}


@dataclass(frozen=True)
class ExpertProfile:
    prior: ExpertPrior
    tokens: int
    top_k: int
    seed: int
    rows_per_expert: tuple[int, ...]

    @property
    def aggregate_rows(self) -> int:
        return sum(self.rows_per_expert)

    @property
    def digest(self) -> str:
        rows = np.asarray(self.rows_per_expert, dtype="<i8")
        return hashlib.sha256(rows.tobytes()).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        return {
            "law": self.prior.value,
            "seed": self.seed,
            "tokens": self.tokens,
            "top_k": self.top_k,
            "aggregate_rows": self.aggregate_rows,
            "rows_sha256": self.digest,
            "rows_per_expert": list(self.rows_per_expert),
        }


def parse_expert_prior(value: str | ExpertPrior) -> ExpertPrior:
    return value if isinstance(value, ExpertPrior) else ExpertPrior(value)


def expert_prior_top_k(value: str | ExpertPrior) -> int:
    prior = parse_expert_prior(value)
    return LEARNED_LAWS[prior].top_k if prior in LEARNED_LAWS else 6


def expert_prior_experts(value: str | ExpertPrior) -> int:
    prior = parse_expert_prior(value)
    return LEARNED_LAWS[prior].experts if prior in LEARNED_LAWS else EXPERTS


def expert_prior_metadata(value: str | ExpertPrior) -> dict[str, object]:
    prior = parse_expert_prior(value)
    if prior is ExpertPrior.DeepSeekHash:
        return {
            "law": prior.value,
            "family": "deepseek",
            "router_kind": "hash",
            "top_k": 6,
            "experts": EXPERTS,
            "coefficients": dict(HASH_COEFFICIENTS),
        }
    law = LEARNED_LAWS[prior]

    def residual_mapping(residual: Residual) -> dict[str, float]:
        return {
            "degrees": residual.degrees,
            "location": residual.location,
            "scale": residual.scale,
            "lower": residual.lower,
            "upper": residual.upper,
        }

    return {
        "law": prior.value,
        "family": LEARNED_FAMILIES[prior],
        "router_kind": "learned",
        "top_k": law.top_k,
        "experts": law.experts,
        "coefficients": {
            "shift": law.shift,
            "head_multiplier": law.head_multiplier,
            "active_a0": law.active[0],
            "active_log_tokens": law.active[1],
            "alpha_b0": law.alpha[0],
            "alpha_log_tokens": law.alpha[1],
            "alpha_active_residual": law.alpha[2],
            "active_residual": residual_mapping(law.active_residual),
            "alpha_residual": residual_mapping(law.alpha_residual),
        },
    }


def default_profile_seed(value: str | ExpertPrior, tokens: int) -> int:
    if tokens < REFERENCE_TOKENS:
        raise ValueError("fitted expert laws require at least 2048 physical tokens")
    prior = parse_expert_prior(value)
    seed = BASE_SEED + tokens * TOKEN_SEED_STRIDE // REFERENCE_TOKENS
    return seed + (HASH_SEED_OFFSET if prior is ExpertPrior.DeepSeekHash else 0)


def _largest_remainder(
    values: np.ndarray, total: int, *, lower: int = 0, upper: int
) -> np.ndarray:
    raw = values * total
    rows = np.clip(np.floor(raw).astype(np.int64), lower, upper)
    while int(rows.sum()) != total:
        add = int(rows.sum()) < total
        candidates = np.flatnonzero(rows < upper if add else rows > lower)
        if not len(candidates):
            raise RuntimeError("constrained rounding cannot preserve routed-row mass")
        errors = raw[candidates] - rows[candidates]
        order = candidates[np.lexsort((candidates, -errors if add else errors))]
        take = min(abs(total - int(rows.sum())), len(order))
        rows[order[:take]] += 1 if add else -1
    return rows


def _active_curve(law: LearnedLaw, active: int, alpha: float) -> np.ndarray:
    values = np.power(np.arange(1, active + 1, dtype=np.float64) + law.shift, -alpha)
    low, high = 0.0, 1.0 / values[-1]
    for _ in range(64):
        middle = (low + high) / 2.0
        if float(np.minimum(1.0, middle * values).sum()) < law.top_k:
            low = middle
        else:
            high = middle
    inclusion = np.minimum(1.0, ((low + high) / 2.0) * values)
    old_head = float(inclusion[0])
    new_head = min(1.0, old_head * law.head_multiplier)
    inclusion[1:] *= (law.top_k - new_head) / (law.top_k - old_head)
    inclusion[0] = new_head
    return inclusion


def _sample_learned(
    law: LearnedLaw, tokens: int, rng: np.random.Generator
) -> np.ndarray:
    experts = law.experts
    x = math.log(tokens / REFERENCE_TOKENS)
    active_residual = law.active_residual.sample(rng)
    active_logit = law.active[0] + law.active[1] * x + active_residual
    active = int(np.rint((experts + 1.0) / (1.0 + math.exp(-active_logit)) - 0.5))
    active = min(experts, max(law.top_k, active))
    alpha_residual = law.alpha_residual.sample(rng)
    alpha = math.exp(
        law.alpha[0]
        + law.alpha[1] * x
        + law.alpha[2] * active_residual
        + alpha_residual
    )
    ranked = _largest_remainder(
        _active_curve(law, active, alpha) / law.top_k,
        law.top_k * tokens,
        upper=tokens,
    )
    rows = np.zeros(experts, dtype=np.int64)
    rows[rng.permutation(experts)[:active]] = ranked
    return rows


def _sample_hash(tokens: int, rng: np.random.Generator) -> np.ndarray:
    ratio = tokens / REFERENCE_TOKENS
    kappa = ratio * (HASH_COEFFICIENTS["kappa_rho"] + 1.0) - 1.0
    mean = HASH_COEFFICIENTS["mu_rho"]
    rho = float(rng.beta(mean * kappa, (1.0 - mean) * kappa))
    body_a = HASH_COEFFICIENTS["a0"] * ratio ** HASH_COEFFICIENTS["beta"]
    probabilities = (1.0 - rho) * rng.dirichlet(np.full(EXPERTS, body_a))
    probabilities[rng.permutation(EXPERTS)[:6]] += rho / 6.0
    return _largest_remainder(probabilities, 6 * tokens, lower=1, upper=tokens)


def sample_expert_profile(
    value: str | ExpertPrior, tokens: int, *, seed: int | None = None
) -> ExpertProfile:
    prior = parse_expert_prior(value)
    selected_seed = default_profile_seed(prior, tokens) if seed is None else int(seed)
    rng = np.random.default_rng(selected_seed)
    rows = (
        _sample_learned(LEARNED_LAWS[prior], tokens, rng)
        if prior in LEARNED_LAWS
        else _sample_hash(tokens, rng)
    )
    top_k = expert_prior_top_k(prior)
    experts = expert_prior_experts(prior)
    if rows.shape != (experts,) or int(rows.sum()) != top_k * tokens:
        raise RuntimeError("fitted expert profile violates its row contract")
    if np.any(rows < 0) or np.any(rows > tokens):
        raise RuntimeError("fitted expert profile violates physical token bounds")
    if prior is ExpertPrior.DeepSeekHash and np.any(rows == 0):
        raise RuntimeError("DeepSeek hash profile must activate every expert")
    return ExpertProfile(prior, tokens, top_k, selected_seed, tuple(map(int, rows)))


def sample_expert_profiles(
    value: str | ExpertPrior, tokens: int, count: int, *, seed: int | None = None
) -> tuple[ExpertProfile, ...]:
    if count <= 0:
        raise ValueError("expert profile count must be positive")
    first_seed = default_profile_seed(value, tokens) if seed is None else int(seed)
    return tuple(
        sample_expert_profile(value, tokens, seed=first_seed + index)
        for index in range(count)
    )


def profile_for_routed_rows(
    value: str | ExpertPrior, aggregate_rows: int
) -> ExpertProfile:
    top_k = expert_prior_top_k(value)
    tokens, remainder = divmod(aggregate_rows, top_k)
    if remainder or tokens <= 0:
        raise ValueError("aggregate rows do not match the selected fitted law")
    return sample_expert_profile(value, tokens)


def profiles_for_routed_rows(
    value: str | ExpertPrior,
    aggregate_rows: int,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[ExpertProfile, ...]:
    top_k = expert_prior_top_k(value)
    tokens, remainder = divmod(aggregate_rows, top_k)
    if remainder or tokens <= 0:
        raise ValueError("aggregate rows do not match the selected fitted law")
    return sample_expert_profiles(value, tokens, count, seed=seed)


@dataclass(frozen=True)
class RouteVector:
    profile: ExpertProfile
    expert_indices: tuple[int, ...]
    expert_offsets: tuple[int, ...]
    group_sizes: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile": self.profile.to_mapping(),
            "expert_indices": list(self.expert_indices),
            "expert_offsets": list(self.expert_offsets),
            "group_sizes": list(self.group_sizes),
        }


def route_bank_for_routed_rows(
    value: str | ExpertPrior,
    aggregate_rows: int,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[RouteVector, ...]:
    vectors = []
    for profile in profiles_for_routed_rows(value, aggregate_rows, count, seed=seed):
        indices = tuple(
            expert for expert, rows in enumerate(profile.rows_per_expert) if rows
        )
        compact_sizes = tuple(profile.rows_per_expert[expert] for expert in indices)
        total = 0
        offsets = []
        for rows in compact_sizes:
            total += rows
            offsets.append(total)
        vectors.append(
            RouteVector(profile, indices, tuple(offsets), profile.rows_per_expert)
        )
    return tuple(vectors)
