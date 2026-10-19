"""Core types. The pedagogical state deliberately has no cultural-schema field (R2.1)."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Move(str, Enum):
    ELICIT = "elicit"                  # diagnostic question
    HINT = "hint"
    WORKED_EXAMPLE = "worked_example"
    EXPLANATION = "explanation"
    ANALOGY = "analogy"
    REFLECT = "reflect"
    ASSESS = "assess"
    REMEDIATE = "remediate_prereq"


# Learning efficacy multiplier and observation informativeness per move.
# These are modelling assumptions for the PoC (stated in the paper's Table 10 / §5.7).
MOVE_EFFICACY = {Move.ELICIT: 0.4, Move.HINT: 0.8, Move.WORKED_EXAMPLE: 1.0, Move.EXPLANATION: 0.9,
                 Move.ANALOGY: 0.8, Move.REFLECT: 0.5, Move.ASSESS: 0.3, Move.REMEDIATE: 1.0}
MOVE_INFO = {Move.ELICIT: 1.0, Move.HINT: 0.5, Move.WORKED_EXAMPLE: 0.3, Move.EXPLANATION: 0.3,
             Move.ANALOGY: 0.5, Move.REFLECT: 0.4, Move.ASSESS: 1.0, Move.REMEDIATE: 0.6}


@dataclass(frozen=True)
class PedState:
    """s_t = (m_t, sigma_t, e_t). The ONLY input the Pedagogy Agent receives about the learner."""
    m: np.ndarray        # mastery in [0,1]^V
    sigma: np.ndarray    # posterior sd of mastery
    e: np.ndarray        # behavioural engagement features: [error_streak, latency_z, help_rate]

    def __hash__(self):  # frozen dataclass with arrays
        return hash((self.m.tobytes(), self.sigma.tobytes(), self.e.tobytes()))


@dataclass(frozen=True)
class Core:
    """Pedagogical core a_ped = (move, target skill, difficulty in [0,1])."""
    move: Move
    skill: int
    difficulty: float


@dataclass
class Schema:
    """Cultural schema c_t: soft, learner-authored, with per-dimension confidence (§4.2.1)."""
    values: dict[str, str]
    confidence: dict[str, float] = field(default_factory=dict)

    def strong(self, min_conf: float = 0.5) -> dict[str, str]:
        return {k: v for k, v in self.values.items() if self.confidence.get(k, 1.0) >= min_conf}


@dataclass
class Item:
    item_id: str
    skill: int
    b: float                      # IRT difficulty (logit scale); d(item) = sigmoid(b)
    stem: str                     # neutral realisation
    claims: tuple[str, ...]       # canonical assessable propositions K(y_ped)
