"""Cultural Adaptation Agent, claim extraction and linguistic load (manuscript §4.2).

y_t = Phi(y_ped, c_t). Backends:
  TemplateCAA   deterministic framing operators per schema dimension (no network; used in tests)
  AnthropicCAA  LLM realisation of Phi via the Anthropic API (set ANTHROPIC_API_KEY)
Claim extraction K(.) runs BEFORE entailment so that framing does not enter the NLI input (R1.4).
"""
from __future__ import annotations
import json, math, os, re

from .types import Item, Schema

CONTEXT = {
    "urban_market": "At a busy city market, a vendor is sorting stock.",
    "rural_farming": "On a farm, a family is dividing the harvest.",
    "cricket": "Before a cricket match, the team is planning the innings.",
    "cooking": "In the kitchen, someone is preparing a recipe for guests.",
    "public_transport": "At the bus depot, a conductor is planning the day's routes.",
}
REGISTER_PREFIX = {"formal_english": "", "simple_english": "Let's try this one. ",
                   "hinglish": "Chalo, let's solve this. "}
COLLAB = {"individual": "", "collaborative": " Work it out together with a partner if you like."}


# ---------------------------------------------------------------- claim extraction
class RegexClaimExtractor:
    """Canonical-proposition extractor for the templated PoC domain. Detects the task
    proposition, any added numbers, and any added scaffolding (hints, partial steps)."""
    PATTERNS = [
        (r"Find (\d+)/(\d+) of (\d+)", "compute {0}/{1} of {2}"),
        (r"What is (\d+)% of (\d+)", "compute {0} percent of {1}"),
        (r"(\d+) items cost (\d+) rupees", "compute {1} divided by {0}"),
        (r"Share (\d+) in the ratio (\d+):(\d+)", "compute first part of {0} in ratio {1}:{2}"),
        (r"rectangle is (\d+) m by (\d+) m", "compute {0} times {1}"),
        (r"interest on (\d+) rupees at (\d+)% per year for (\d+) years",
         "compute simple interest on {0} at {1} percent for {2} years"),
    ]
    SCAFFOLD = re.compile(r"\b(hint|first,? (divide|multiply)|step 1|the answer is|start by)\b", re.I)

    def extract(self, text: str) -> tuple[str, ...]:
        claims = []
        for pat, tmpl in self.PATTERNS:
            for m in re.finditer(pat, text):
                claims.append(tmpl.format(*m.groups()))
        if self.SCAFFOLD.search(text):
            claims.append("scaffold: " + self.SCAFFOLD.search(text).group(0).lower())
        return tuple(sorted(set(claims)))


class AnthropicClaimExtractor:
    """LLM claim extraction for free-text items: returns short canonical propositions."""
    PROMPT = ("List every assessable claim, task requirement, given quantity and any hint or partial "
              "solution in the text below as short canonical propositions. Ignore story framing, names "
              "and setting. Reply with a JSON array of strings only.\n\nTEXT:\n{t}")

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def extract(self, text: str) -> tuple[str, ...]:
        r = self.client.messages.create(model=self.model, max_tokens=400,
                                        messages=[{"role": "user", "content": self.PROMPT.format(t=text)}])
        raw = r.content[0].text.strip().strip("`").removeprefix("json").strip()
        return tuple(sorted(set(json.loads(raw))))


# ---------------------------------------------------------------- CAA backends
class TemplateCAA:
    """Applies framing operators for the strongest schema dimensions. `latitude` in (0,1]
    controls how many dimensions are applied; the orchestrator lowers it on each retry."""
    name = "template"

    def adapt(self, item: Item, schema: Schema, latitude: float = 1.0) -> str:
        dims = sorted(schema.strong().items(), key=lambda kv: -schema.confidence.get(kv[0], 1))
        k = max(0, math.ceil(latitude * len(dims)))
        use = dict(dims[:k])
        text = item.stem
        if "referential_context" in use:
            text = CONTEXT.get(use["referential_context"], "") + " " + text
        if use.get("framing") == "definition_first":
            text = text + " Use the definition you learned."
        text = REGISTER_PREFIX.get(use.get("language_register", ""), "") + text
        text = text + COLLAB.get(use.get("collaboration", ""), "")
        return text.strip()


class AnthropicCAA:
    """LLM realisation of Phi. The prompt forbids changing quantities, adding hints or steps."""
    name = "anthropic"
    PROMPT = ("Rewrite this maths task for a learner. Change ONLY framing, setting, names and register "
              "to fit the learner profile. Keep every number, the question asked, and the difficulty "
              "identical. Do not add hints, steps, simplifications or the answer. Adaptation latitude: "
              "{lat:.2f} (1 = full adaptation, 0 = minimal).\nLearner profile: {prof}\nTask: {stem}\n"
              "Reply with the rewritten task only.")

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def adapt(self, item: Item, schema: Schema, latitude: float = 1.0) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=300,
            messages=[{"role": "user", "content": self.PROMPT.format(lat=latitude, prof=json.dumps(schema.strong()),
                                                                     stem=item.stem)}])
        return r.content[0].text.strip()


# ---------------------------------------------------------------- linguistic load
def _syllables(w: str) -> int:
    w = w.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def linguistic_load(text: str) -> dict:
    """Proxy for language-driven cognitive load (Sweller 1988; Abedi & Lord 2001).
    Returns the components and a normalised scalar used by the difficulty-invariance check."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    nw, ns = max(len(words), 1), max(len(sents), 1)
    fk = 0.39 * nw / ns + 11.8 * sum(_syllables(w) for w in words) / nw - 15.59
    long_ratio = sum(len(w) >= 8 for w in words) / nw
    referents = len(re.findall(r"\d+", text)) + len(re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", text))
    scalar = 0.5 * max(fk, 0) / 12 + 0.3 * (nw / ns) / 25 + 0.2 * long_ratio
    return {"fk_grade": fk, "words_per_sentence": nw / ns, "long_word_ratio": long_ratio,
            "referents": referents, "load": scalar}
