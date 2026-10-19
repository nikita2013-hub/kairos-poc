"""Content-invariance verifier (manuscript §4.2.2, R1.4).

Pass iff (i) claims of y_ped and y_t mutually entail (bidirectional NLI on extracted claims),
and (ii) the linguistic-load increase is within eta_L. Any neutral/contradiction verdict counts
as "not verified" (errors cost responsiveness, not content).
Backends: ExactClaimNLI (normalised string equality; deterministic, for tests) and
HFNLI (Hugging Face cross-encoder, e.g. microsoft/deberta-large-mnli).
"""
from __future__ import annotations
from dataclasses import dataclass, field


class ExactClaimNLI:
    name = "exact"

    def entails(self, premise: str, hypothesis: str) -> float:
        prem = {c.strip().lower() for c in premise.split(" ; ")}
        return 1.0 if hypothesis.strip().lower() in prem else 0.0


class HFNLI:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent_idx = labels.get("entailment", 2)
        self.device = device

    def entails(self, premise: str, hypothesis: str) -> float:
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)[0]
        return float(probs[self.ent_idx]) if int(probs.argmax()) == self.ent_idx else 0.0


@dataclass
class Verdict:
    ok: bool
    reasons: list = field(default_factory=list)
    forward: float = 1.0
    backward: float = 1.0
    load_delta: float = 0.0


class ContentVerifier:
    def __init__(self, nli, extractor, eta_load: float = 0.15, threshold: float = 0.5):
        self.nli, self.extractor, self.eta, self.threshold = nli, extractor, eta_load, threshold

    def verify(self, neutral_text: str, adapted_text: str, neutral_claims=None) -> Verdict:
        from .presentation import linguistic_load
        cn = tuple(neutral_claims) if neutral_claims else self.extractor.extract(neutral_text)
        ca = self.extractor.extract(adapted_text)
        reasons = []
        if not ca:
            return Verdict(False, ["no_claims_extracted"], 0.0, 0.0)
        pa, pn = " ; ".join(ca), " ; ".join(cn)
        fwd = min(self.nli.entails(pa, c) for c in cn) if cn else 1.0     # nothing omitted / softened
        bwd = min(self.nli.entails(pn, c) for c in ca)                    # nothing added (e.g. hints)
        if fwd < self.threshold: reasons.append("omission_or_change")
        if bwd < self.threshold: reasons.append("addition")
        dl = linguistic_load(adapted_text)["load"] - linguistic_load(neutral_text)["load"]
        if dl > self.eta: reasons.append("linguistic_load")
        return Verdict(not reasons, reasons, fwd, bwd, dl)
