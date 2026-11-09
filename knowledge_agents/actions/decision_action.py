# -*- coding: utf-8 -*-
"""Core action of the decision agent: consumes BackgroundKnowledge + ComputedKnowledge and emits the final Answer.

Scope is limited to pairwise / membership. Two decision kernels (env SLC_DECISION_MODE=heuristic|llm|auto):
  - heuristic (default, environment-independent, deterministic, easy to evaluate): transparent weighted scoring of the candidate genes
  - llm/auto: ask the MetaGPT LLM to synthesise the evidence into an answer; on timeout/or failure it falls back automatically to heuristic

Evidence-integrity invariant: a candidate with no usable evidence -> abstains (abstained), and never treats "unknown" as negative, nor picks a winner at random.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re

from metagpt.actions import Action

from knowledge_agents.schema import (
    Answer,
    BackgroundKnowledge,
    ComputedKnowledge,
    ToolStatus,
)
from knowledge_agents.graph_features import (
    collect_path_rows, feat_vec, gene_graph_feat,
)

_P_VALUE_RE = re.compile(r"p[\s_-]*value[:\s=]*([0-9.eE+-]+)", re.I)

# Weights of the evidence channels (transparent and tunable; not a biological prior, only a ranking heuristic)
W_KG_DISEASE_PATH = 0.5     # per matched gene-event-disease path
W_DIRECTION = 0.6           # temporal-causal direction(rel_AB x rel_BC)adds/when it agrees with the question verb and/ (signed)
W_KG_NEIGHBOR = 0.05        # per matched graph neighbour (family/interaction, etc.; weak signal)
W_PPR = 1.0                 # network proximity after normalisation (the main gene<->disease relation signal)
W_PPR_RANK = 0.3            # extra ranking bonus once the gene enters the disease PPR candidate list
W_KM = 0.5                  # -log10(p) after clipping
W_DE = 0.25                 # |log2fc| after clipping
W_TIMER = 0.15             # TIMER: non-zero whenever expression evidence is available (table or plot)
W_BUFFER = 0.8             # paralog buffering: how the redundancy value in [0, 1] scales the one-sided
                           # weakening of an essential/positive polarity (weighted nearly on par with the causal direction)
MIN_SCORE_TO_ANSWER = 0.5   # If the total evidence score of a single gene is below this -> abstain
NORM_STOP = frozenset({"cancer", "cancers", "tumor", "tumour", "syndrome", "carcinoma", "disease", "diseases"})

# Conformal thresholds are cached at module level (None = not loaded, {} = disabled or no file). They cannot be declared as pydantic Action class attributes.
_CONFORMAL_CACHE: dict | None = None


def _sign_of(effect) -> int:
    """composite sign gene_disease_effect -> +1 PROMOTION / -1 SUPPRESSION / 0 unknown."""
    e = str(effect or "").upper()
    return 1 if e == "PROMOTION" else (-1 if e == "SUPPRESSION" else 0)


# ---------------------------------------------------------------- pairwise graph-feature logistic
# Cached by the resolved artifact path: an unset SLC_PAIRWISE_MODEL leaves it None (the transparent
# weighted rules alone); when it is set and the file exists, the artifact is loaded.
# Inference only applies the stored coefficients (linear + sigmoid), with no sklearn dependency, so
# the online environment cannot drift.
_PAIRWISE_MODEL_CACHE: dict[str, dict | None] = {}


def _pairwise_model() -> dict | None:
    path = os.environ.get("SLC_PAIRWISE_MODEL", "") or ""
    if not path:
        return None
    if path in _PAIRWISE_MODEL_CACHE:
        return _PAIRWISE_MODEL_CACHE[path]
    model: dict | None = None
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("task") == "pairwise_graph_logistic" and m.get("coef"):
                model = m
        except Exception:  # noqa BLE001 - a broken artifact falls back to the rules and never blocks the decision
            model = None
    _PAIRWISE_MODEL_CACHE[path] = model
    return model


def _score_pairwise_graph(kg: BackgroundKnowledge, candidates: list[str],
                          cancer: str) -> tuple[list[str], dict[str, float], dict] | None:
    """Score each candidate gene with the fitted graph-feature logistic model as P(winner).
    Returns (candidates in descending score order, {gene: prob}, model). It returns None - so the
    caller falls back to the transparent rules and fabricates nothing - when no model artifact is
    configured or when neither candidate has gene_disease_paths graph evidence."""
    model = _pairwise_model()
    if not model or len(candidates) < 2:
        return None
    rows = collect_path_rows(kg)
    feats = {g: gene_graph_feat(g, rows, cancer) for g in candidates}
    if all(feats[g].get("n_paths", 0) == 0 for g in candidates):
        return None                                   # no evidence anywhere on the graph chain -> fall back honestly to the rules
    cols = model["features"]
    mu, sd, coef = model["mu"], model["sd"], model["coef"]
    b = float(model.get("intercept", 0.0))
    probs: dict[str, float] = {}
    for g in candidates:
        x = feat_vec(feats[g], cols)
        lg = b + sum(coef[i] * ((x[i] - mu[i]) / sd[i]) for i in range(len(cols)))
        probs[g] = 1.0 / (1.0 + math.exp(-lg))
    ranked = sorted(candidates, key=lambda g: (-probs[g], g))
    return ranked, probs, model


class AnswerDecision(Action):
    """Decision action: produces Answer."""

    name: str = "AnswerDecision"

    DECISION_PROMPT: str = """You are making an evidence-synthesis decision for an SLC-gene CRISPR screen.
Use only the evidence given below and invent nothing.
Evidence integrity: a channel reported as unavailable, unknown or rejected reads as "unknown" and
never as a negative result.
Synthesis principle: the causal direction (gene_disease_effect, decomposed from a RelaEvent
rel_AB x rel_BC chain) and the computed magnitudes (PPR network proximity, TIMER expression, KM
survival, DESeq2 differential expression) are all reference knowledge. Weigh them together and let
no single channel dictate the answer: the direction supplies the promotion/suppression polarity,
the magnitude supplies the strength of relevance, and when the two conflict you must reflect that
honestly in the confidence value.
Note: a disease association, a graph proximity or a differential-expression signal is not CRISPR
loss-of-function essentiality. Never read "strongly associated" as a positive on its own.
Paralog buffering: if the candidate gene has many co-expressed paralogues from the same subfamily
in the same cancer type (high redundancy), its knockout is likely buffered by those paralogues, so
the gene looks less essential. This is one-sided evidence that weakens a positive call; the absence
of paralogues does not mean the gene is essential.

Question (bucket={bucket}): {question}
Candidate genes: {candidates}
Question constraints: {context}

[Background knowledge: graph + literature, with causal direction]
{kg_json}

[Computed knowledge: PPR / TIMER / KM / DE]
{ck_json}

Output exactly one JSON object:
- pairwise:   {{"winner": "<one of the candidates or null>", "confidence": 0~1, "abstained": true/false, "rationale": "..."}}
- membership: {{"answer": true/false/null, "confidence": 0~1, "abstained": true/false, "rationale": "..."}}
When the evidence is insufficient, set winner/answer to null and set abstained to true.
"""

    # Used when the target cancer type carries no direct evidence: the model reasons about how the
    # neighbouring cancer types transfer to the target before answering.
    ANALOGY_PROMPT: str = """You are answering an SLC-gene CRISPR screen question. For the target cancer type
{target_cancer}, the candidate genes lack direct graph proximity or omics evidence. Do not treat
missing evidence as a negative result; reason across cancer types instead.
Weigh the causal direction (gene_disease_effect) and the computed magnitudes (PPR / TIMER / KM / DE)
together as reference knowledge; no single channel dictates. An association, a proximity or an
expression signal is not CRISPR essentiality.

Below is every piece of evidence these genes already carry, including their links to other cancer
types and diseases; treat all of it as knowledge:
[Background knowledge: graph + literature]
{kg_brief}
[Computed knowledge: PPR / TIMER / KM / DE]
{ck_brief}

Neighbouring cancers this gene is known to be linked to: {related_cancers}
Target cancer type: {target_cancer}   Question bucket: {bucket}   Candidate genes: {candidates}
Question: {question}

Reason about how far these neighbouring cancers transfer to the target in tissue of origin, cell
lineage, molecular subtype and pathway dependence, then judge the most likely direction of effect of
this candidate gene in the target cancer type. You MUST give one explicit answer; abstention is not
allowed here:
- pairwise: choose one of the two candidates, whichever the evidence and the analogy favour
- membership: true or false
Output only the JSON:
- pairwise:   {{"winner": "<one of the candidates>", "confidence": 0~1, "abstained": false, "analogy": "<neighbouring-to-target reasoning>", "rationale": "..."}}
- membership: {{"answer": true/false, "confidence": 0~1, "abstained": false, "analogy": "<neighbouring-to-target reasoning>", "rationale": "..."}}
The confidence must reflect the uncertainty of the cross-cancer analogy; it is usually lower than
for a decision backed by direct evidence.
"""

    @staticmethod
    def _km_minuslog10(stats_text: str | None) -> float:
        if not stats_text:
            return 0.0
        m = _P_VALUE_RE.search(stats_text)
        if not m:
            return 0.0
        try:
            p = float(m.group(1))
        except ValueError:
            return 0.0
        if p <= 0:
            return 20.0
        return min(20.0, -math.log10(p))

    @classmethod
    def _collect_scores(
        cls, kg: BackgroundKnowledge, ck: ComputedKnowledge, candidates: list[str], verb: int = 0
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, set[str]]]:
        """Returns (strength, pol, buf, basis):
          strength[g] = total evidence per channel from magnitude/association (unsigned, >=0);
          pol[g]      = signed net vote of the temporal-causal direction sum(verb x sign), contributed only by gene_disease_effect;
          buf[g]      = paralog-buffering polarity in [-1, 0] (minus the redundancy); the higher it is, the more it pushes "essential/positive" towards negative. One-sided, and recorded as 0 when there is no data;
          basis[g]    = labels of the channels that fired.
        Direction and magnitude are accumulated separately so that unsigned proximity/expression are never mistaken for polarity evidence."""
        strength: dict[str, float] = {g: 0.0 for g in candidates}
        pol: dict[str, float] = {g: 0.0 for g in candidates}
        buf: dict[str, float] = {g: 0.0 for g in candidates}
        basis: dict[str, set[str]] = {g: set() for g in candidates}

        def bump(gene: str, val: float, tag: str):
            if gene in strength:
                strength[gene] += val
                basis[gene].add(tag)

        def bump_pol(gene: str, signed: float):
            if gene in pol:
                pol[gene] += signed

        # 1) KG disease paths / neighbours (from BackgroundKnowledge.evidence)
        for ev in kg.evidence:
            for row in ev.rows or []:
                if not isinstance(row, dict):
                    continue
                g = str(row.get("gene", "")).upper()
                if ev.operation == "gene_disease_paths":
                    bump(g, W_KG_DISEASE_PATH, "kg_disease_path")   # association strength (magnitude, unsigned)
                    if verb:                                        # causal direction, recorded separately in poland never in strength)
                        s = _sign_of(row.get("gene_disease_effect"))
                        if s:
                            bump_pol(g, verb * s)
                            basis[g].add("kg_direction" if verb * s > 0 else "kg_direction_against")
                elif ev.operation == "gene_neighbors":
                    bump(g, W_KG_NEIGHBOR, "kg_neighbor")

        # 2) PPR proximity (normalised to a relative value inside the candidate set, plus + on-list rank bonus)
        if ck.graph_computation and ck.graph_computation.status is ToolStatus.OK:
            cands = ck.graph_computation.candidates or []
            vals = {c.gene.upper(): (c.degree_corrected_score or 0.0) for c in cands}
            mx = max(vals.values(), default=0.0) or 1.0
            n = max(1, len(cands))
            for idx, c in enumerate(cands):
                g = c.gene.upper()
                bump(g, W_PPR * ((c.degree_corrected_score or 0.0) / mx), "ppr")
                bump(g, W_PPR_RANK * (1.0 - idx / n), "ppr_rank")  # the higher it ranks, the more it looks like a disease-related gene

        # 3) TIMER expression evidence (a table or a box plot counts as a weak positive)
        expr_map = ck.expression_by_gene or ({"_": ck.expression} if ck.expression else {})
        for gene, ex in expr_map.items():
            if gene == "_" or not ex or ex.status is not ToolStatus.OK:
                continue
            has_evidence = bool(ex.rows) or bool(ex.boxplot_path) or (ex.row_count or 0) > 0
            if has_evidence:
                bump(gene.upper(), W_TIMER, "timer")

        # 4) KM survival significance (per gene)
        surv = ck.survival_by_gene or ({"_": ck.survival} if ck.survival else {})
        for gene, km in surv.items():
            if gene == "_" or not km or km.status is not ToolStatus.OK:
                continue
            bump(gene.upper(), W_KM * (cls._km_minuslog10(km.stats_text) / 10.0), "km")

        # 5) DESeq2 differential expression (cancer-type level, only for candidates that hit top_genes)
        de = ck.differential_expression
        if de and de.status is ToolStatus.OK:
            for row in de.top_genes or []:
                g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
                l2 = row.get("log2fc")
                if l2 is not None:
                    bump(g, W_DE * min(abs(float(l2)), 5.0) / 5.0, "de")

        # 6) paralog buffering (paralog redundancy -> weakens the essentiality polarity; one-sided and never contributes to strength)
        for gene_u, red in (ck.paralog_redundancy or {}).items():
            g = gene_u.upper()
            if g in buf:
                r = max(0.0, min(1.0, float(red or 0.0)))
                buf[g] = -r                      #  in  [-1,0]
                if r > 0:
                    basis[g].add("paralog_buffered")
                    bump(g, 0.0, "paralog_redundancy")  # recorded in basis, not in strength
                else:
                    basis[g].add("paralog_solo")

        return strength, pol, buf, basis

    # ------------------------------------------------------------------ Temporal-causal direction (rel_AB  x  rel_BC composite sign)
    @classmethod
    def _norm_terms(cls, text: str) -> set[str]:
        return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
                if len(w) > 3 and w not in NORM_STOP}

    @staticmethod
    def _verb_dir(task_type: str) -> int:
        t = (task_type or "").lower()
        if t.endswith("positive"):
            return 1    # promote / meaning: the expected gene->disease = PROMOTION
        if t.endswith("negative"):
            return -1   # inhibit / meaning: the expected gene->disease = SUPPRESSION
        return 0

    @staticmethod
    def _direction(kg: BackgroundKnowledge, gene: str, cancer: str) -> dict:
        """Count, through RelaEvent, the RelaEvent composite gene_disease_effect of promotion/suppression events for this gene.
        Disease rows matching the question cancer type come first(matched=True); if no cancer type matches, fall back to all diseases(matched=False)."""
        terms = AnswerDecision._norm_terms(cancer)
        p_m = s_m = p_a = s_a = 0
        for ev in kg.evidence:
            if ev.operation != "gene_disease_paths":
                continue
            for row in ev.rows or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("gene", "")).upper() != gene.upper():
                    continue
                eff = str(row.get("gene_disease_effect") or "").upper()
                if eff not in ("PROMOTION", "SUPPRESSION"):
                    continue
                hit = bool(terms) and bool(terms & AnswerDecision._norm_terms(str(row.get("disease", ""))))
                if eff == "PROMOTION":
                    p_a += 1
                    p_m += 1 if hit else 0
                else:
                    s_a += 1
                    s_m += 1 if hit else 0
        if p_m + s_m > 0:
            promo, supp, matched = p_m, s_m, True
        elif p_a + s_a > 0:
            promo, supp, matched = p_a, s_a, False
        else:
            return {"has": False, "net": 0, "promo": 0, "supp": 0, "matched": False}
        return {"has": True, "net": promo - supp, "promo": promo, "supp": supp, "matched": matched}

    # ------------------------------------------------------------------ heuristic
    def _heuristic(self, question: str, kg: BackgroundKnowledge, ck: ComputedKnowledge,
                   bucket: str, candidates: list[str], task_type: str, context: dict | None = None) -> Answer:
        """Deterministic ablation baseline: there is no direction-first short circuit; the causal
        direction (polarity) and the magnitude (strength) are weighed together before deciding.
        The direction only supplies polarity and the magnitude only supplies strength or ranking;
        neither one vetoes the other on its own."""
        cancer = (context or {}).get("cancer", "") or self._infer_cancer(question)
        verb = self._verb_dir(task_type)
        strength, pol, buf, basis = self._collect_scores(kg, ck, candidates, verb)

        if bucket == "pairwise":
            # Prefer the pairwise graph-feature logistic artifact selected by SLC_PAIRWISE_MODEL.
            # When it is disabled, or when neither candidate has evidence along the
            # SLC -> pathway -> disease chain (so the model cannot rank), fall back honestly to the
            # transparent weighted rules below and never invent a winner.
            gp = _score_pairwise_graph(kg, candidates, cancer)
            if gp is not None:
                ranked, probs, model = gp
                best = ranked[0]
                second = ranked[1] if len(ranked) > 1 else None
                gap = probs[best] - (probs[second] if second else probs[best])
                ho = (model.get("holdout") or {}).get("acc")
                ho_txt = f"{ho:.0%}" if isinstance(ho, (int, float)) else "?"
                return self._make(bucket, task_type, candidates, strength, basis,
                                  winner=best, answer=None, abstained=False,
                                  confidence=round(max(0.0, min(1.0, probs[best])), 3),
                                  rationale=f"[pairwise graph-feature logistic model] {best} (P(winner)={probs[best]:.2f}) vs "
                                            f"{second} (P={probs.get(second, 0):.2f}), gap={gap:.2f}; "
                                            f"ranking follows the SLC -> pathway -> disease decomposed graph features "
                                            f"fed to a learned logistic model (held-out accuracy ~{ho_txt}). "
                                            f"This is an association-direction proxy, not a CRISPR readout.")
            # Combined: magnitude + signed direction + paralog buffering (redundancy pushes the
            # candidate towards the less-essential side).
            combined = {g: strength[g] + W_DIRECTION * pol[g] + W_BUFFER * buf[g] for g in candidates}
            ranked = sorted(candidates, key=lambda g: (-combined[g], g))
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            if strength[best] < MIN_SCORE_TO_ANSWER and pol[best] == 0 and buf[best] == 0:
                return self._make(bucket, task_type, candidates, strength, basis,
                                  winner=None, answer=None, abstained=True, confidence=0.0,
                                  rationale=f"Direction, magnitude and paralog-buffering evidence are all short of the floor (best={best} strength={strength[best]:.2f}, pol=0, buf=0); abstaining instead of fabricating.")
            gap = combined[best] - (combined[second] if second else combined[best])
            conf = max(0.0, min(1.0, 0.5 * math.tanh(gap) + 0.1 * min(1.0, strength[best])))
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=best, answer=None, abstained=False, confidence=round(conf, 3),
                              rationale=f"[direction+magnitude+buffering combined] best={best}(combined={combined[best]:.2f} pol={pol[best]:+.0f} "
                                        f"buf={buf[best]:+.2f} strength={strength[best]:.2f}) vs {second}(combined={combined.get(second, 0):.2f}), "
                                        f"Evidence: {sorted(basis[best])}.")

        # membership
        g = candidates[0] if candidates else ""
        p = pol.get(g, 0.0)
        b = buf.get(g, 0.0)
        s = strength.get(g, 0.0)
        # R2 (gate SLC_MEM_R2=1): membership switches to essentiality / loss-of-function evidence
        # re-ranking and drops the contradictory promotion/suppression polarity. Off by default, so
        # the logic below stays exactly as the frozen release produced it.
        if os.environ.get("SLC_MEM_R2", "0") == "1":
            return self._membership_r2(bucket, task_type, candidates, strength, basis, g, s, verb)
        # Key correction: the sign of a membership answer is set by the causal direction p alone.
        # Paralog buffering can only WEAKEN the net strength of an existing direction; it must never
        # call a gene negative when no causal direction is present (unknown != negative).
        # The old rule added the buffer to a zero direction: net=p+W_BUFFER*b so with p=0 a buffer alone(net=0.8*b<0)was enough to mislabel many genes as negative -> false negatives in bulk.
        if p == 0:
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=None, answer=None, abstained=True, confidence=0.0,
                              rationale=f"{g} has only magnitude/association evidence(strength={s:.2f}, channels{sorted(basis.get(g, set()))});"
                                        f"with no causal polarity direction(rela promotion/suppression unestablished), so paralog buffering alone cannot decide negative; stays unknown and abstains honestly.")
        net = p + W_BUFFER * b            # With a direction present: the buffer only weakens the net strength of that direction (one-sided, mostly weakening positives)
        if abs(net) < 1e-9:
            # The buffer cancels the net polarity exactly to 0 -> cannot orient, so abstain honestly
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=None, answer=None, abstained=True, confidence=0.0,
                              rationale=f"{g} the causal direction was cancelled exactly by paralog buffering(net=0), not enough to decide, stays unknown.")
        ans = net > 0
        conf = max(0.0, min(1.0, 0.3 + 0.12 * abs(net) + 0.1 * math.log1p(s)))
        return self._make(bucket, task_type, candidates, strength, basis,
                          winner=None, answer=ans, abstained=False, confidence=round(conf, 3),
                          rationale=f"[direction sets polarity+buffer weakens] {g} net polarity net={net:+.2f}"
                                    f"(causal pol={p:+.0f} and paralog buffering buf={b:+.2f}x{W_BUFFER}; verb{'promote' if verb > 0 else 'inhibit' if verb < 0 else '?'});"
                                    f"magnitude strength={s:.2f} channels{sorted(basis.get(g, set()))} -> {'positive' if ans else 'negative'}"
                                    f"(polarity comes only from the causal direction, the buffer only weakens and the magnitude sets the confidence; "
                                    f"all of it is knowledge-graph and omics evidence, not a CRISPR readout.")
    
    def _membership_r2(self, bucket, task_type, candidates, strength, basis, g, s, verb) -> Answer:
        """R2 re-frames membership from "polarity decides the answer" to "essentiality /
        loss-of-function evidence re-ranking". Contradictory promotion versus suppression polarities
        in the literature are no longer evidence for the answer; the predictors are polarity conflict
        (towards essential), paralogue orphan (towards essential), paralog buffering (towards
        non-essential) and magnitude (towards essential). The coefficients were fitted by 5-fold
        cross-validation on the n=53 membership subsample (held-out accuracy 66.8%, above the 60.4%
        majority floor). This changes only the membership decision and never abstains, which helps
        overall accuracy; the pairwise and gate-off paths are completely unaffected."""
        tags = basis.get(g, set())
        conflict = 1.0 if {"kg_direction", "kg_direction_against"} <= tags else 0.0
        dir_present = 1.0 if (tags & {"kg_direction", "kg_direction_against"}) else 0.0
        solo = 1.0 if "paralog_solo" in tags else 0.0
        buffered = 1.0 if "paralog_buffered" in tags else 0.0
        mag = math.tanh(s)
        z = (1.732673 * conflict - 1.733301 * dir_present + 2.634866 * solo
             - 0.619402 * buffered + 1.934390 * mag - 0.408812)
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        ess = prob > 0.5
        return self._make(bucket, task_type, candidates, strength, basis,
                          winner=None, answer=ess, abstained=False, confidence=round(prob, 3),
                          rationale=f"[R2 essentiality re-ranking] {g} polarity conflict={conflict:.0f} direction evidence={dir_present:.0f} "
                                    f"paralogue orphan={solo:.0f} buffered redundancy={buffered:.0f} magnitude={s:.2f} -> "
                                    f"P(essential)={prob:.3f} -> {'essential' if ess else 'non-essential'} "
                                    f"(the construct is CRISPR loss-of-function essentiality, not a "
                                    f"promotion/suppression direction; verb "
                                    f"{'promote' if verb > 0 else 'inhibit' if verb < 0 else '?'}).")

    def _make(self, bucket, task_type, candidates, scores, basis, *, winner, answer,
              abstained, confidence, rationale) -> Answer:
        flat_basis = sorted({t for tags in basis.values() for t in tags})
        return Answer(
            question=f"bucket={bucket} candidates={candidates}",
            status=ToolStatus.OK if not abstained else ToolStatus.UNAVAILABLE,
            summary=rationale,
            bucket=bucket, task_type=task_type, candidates=list(candidates),
            winner=winner, answer=answer, abstained=abstained,
            confidence=float(confidence),
            gene_scores={k: round(v, 4) for k, v in scores.items()},
            basis=flat_basis, rationale=rationale,
            decision_mode="heuristic", used_llm=False,
        )

    # ------------------------------------------------------------------ llm
    async def _llm(self, question, kg, ck, bucket, candidates, task_type, context) -> Answer | None:
        kg_json = kg.model_dump_json(indent=2)[:6000]
        ck_json = ck.model_dump_json(indent=2)[:6000]
        prompt = self.DECISION_PROMPT.format(
            bucket=bucket, question=question, candidates=candidates,
            context=json.dumps(context, ensure_ascii=False), kg_json=kg_json, ck_json=ck_json,
        )
        try:
            import asyncio
            rsp = await asyncio.wait_for(self._aask(prompt), timeout=float(os.environ.get("SLC_DECISION_LLM_TIMEOUT", "30")))
        except Exception:  # noqa BLE001 - LLM falls back when unavailable and never blocks the pipeline
            return None
        data = self._extract_json(rsp)
        if not data:
            return None
        abstained = bool(data.get("abstained")) or (data.get("winner") is None and data.get("answer") is None)
        strength, _pol, _buf, basis = self._collect_scores(kg, ck, candidates, self._verb_dir(task_type))
        return self._make_from_llm(bucket, task_type, candidates, strength, basis, data, abstained)

    def _make_from_llm(self, bucket, task_type, candidates, scores, basis, data, abstained,
                       analogy_cancers=None) -> Answer:
        flat_basis = sorted({t for tags in basis.values() for t in tags})
        winner = data.get("winner") if bucket == "pairwise" else None
        answer = data.get("answer") if bucket == "membership" else None
        if winner not in candidates:
            winner = None
        rationale = str(data.get("rationale", ""))
        if data.get("analogy"):
            rationale = f"[cross-cancer analogy] {data['analogy']} | {rationale}"
        return Answer(
            question=f"bucket={bucket} candidates={candidates}",
            status=ToolStatus.OK if not abstained else ToolStatus.UNAVAILABLE,
            summary=rationale,
            bucket=bucket, task_type=task_type, candidates=list(candidates),
            winner=winner, answer=answer, abstained=abstained,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            gene_scores={k: round(v, 4) for k, v in scores.items()},
            basis=flat_basis, rationale=rationale,
            analogy_cancers=list(analogy_cancers or []),
            decision_mode="llm", used_llm=True,
        )

    # ------------------------------------------------------------------ cross-cancer analogy
    @staticmethod
    def _related_cancers(kg: BackgroundKnowledge, candidates: list[str]) -> list[str]:
        """Extract, from the KG gene_disease_paths neighbouring cancers/diseases already linked to the candidate genes."""
        wanted = {c.upper() for c in candidates}
        cancers: list[str] = []
        for ev in kg.evidence:
            if ev.operation != "gene_disease_paths":
                continue
            for row in ev.rows or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("gene", "")).upper() not in wanted and wanted:
                    continue
                d = row.get("disease") or row.get("disease_name") or row.get("name")
                if d and str(d) not in cancers:
                    cancers.append(str(d))
        return cancers[:20]

    @staticmethod
    def _evidence_brief(obj) -> str:
        """Compress kg/ck into short text for the LLM (only non-empty receipts, bounded length)."""
        lines: list[str] = []
        evs = getattr(obj, "evidence", None)
        if evs is not None:  # BackgroundKnowledge
            for ev in evs:
                if ev.rows:
                    lines.append(f"- {ev.operation}: {json.dumps(ev.rows[:4], ensure_ascii=False, default=str)}")
        else:  # ComputedKnowledge
            gc = obj.graph_computation
            if gc and gc.status is ToolStatus.OK and gc.candidates:
                top = [f"{c.gene}={c.degree_corrected_score}" for c in gc.candidates[:8]]
                lines.append(f"- PPRdisease proximityTop: {top}")
            for g, ex in (obj.expression_by_gene or {}).items():
                lines.append(f"- TIMER[{g}]: status={ex.status.value} rows={ex.row_count} scope={ex.evidence_scope}")
            for g, km in (obj.survival_by_gene or {}).items():
                lines.append(f"- KM[{g}]: status={km.status.value} stats={ (km.stats_text or '')[:60]!r}")
            de = obj.differential_expression
            if de and de.status is ToolStatus.OK:
                lines.append(f"- DESeq2({de.cancer_type}): significant={de.significant_count}")
            red = getattr(obj, "paralog_redundancy", None) or {}
            pmap = getattr(obj, "paralog_partners", None) or {}
            for g, r in list(red.items())[:8]:
                if r and r > 0:
                    lines.append(f"- paralogue redundancy[{g}]: redundancy={r:.2f} co-expressed paralogues in the same subfamily={ (pmap.get(g) or [])[:6] }"
                                 f"(high redundancy=knockout is easily buffered=more likely non-essential/negative)")
        return "\n".join(lines) or "(no notable evidence)"

    async def _llm_analogy(self, question, kg, ck, bucket, candidates, task_type, context):
        target_cancer = (context or {}).get("cancer", "") or self._infer_cancer(question)
        related = self._related_cancers(kg, candidates)
        prompt = self.ANALOGY_PROMPT.format(
            target_cancer=target_cancer or "not specified",
            kg_brief=self._evidence_brief(kg), ck_brief=self._evidence_brief(ck),
            related_cancers=related or "(no linked cancer types inKG; reason from cancer-biology common sense)",
            bucket=bucket, candidates=candidates, question=question,
        )
        try:
            rsp = await asyncio.wait_for(self._aask(prompt),
                                         timeout=float(os.environ.get("SLC_DECISION_LLM_TIMEOUT", "40")))
        except Exception:  # noqa BLE001 - LLM returns Nonewhen unavailable and run() handles the fallback
            return None
        data = self._extract_json(rsp)
        if not data:
            return None
        strength, _pol, _buf, basis = self._collect_scores(kg, ck, candidates, self._verb_dir(task_type))
        winner = data.get("winner") if bucket == "pairwise" else None
        answer = data.get("answer") if bucket == "membership" else None
        abstained = bool(data.get("abstained")) or (
            (bucket == "pairwise" and (not winner or winner.upper() not in {c.upper() for c in candidates}))
            or (bucket == "membership" and answer is None))
        return self._make_from_llm(bucket, task_type, candidates, strength, basis, data, abstained,
                                   analogy_cancers=related)

    @staticmethod
    def _infer_cancer(question: str) -> str:
        m = re.search(r"cancer=([A-Za-z0-9_ \-]+?)(?:\s+bucket=|\s+Under|$)", question)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_json(rsp: str) -> dict | None:
        m = re.search(r"\{.*\}", rsp, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    async def run(
        self,
        question: str,
        *,
        kg: BackgroundKnowledge,
        ck: ComputedKnowledge,
        bucket: str,
        candidates: list[str],
        task_type: str = "",
        context: dict | None = None,
        prior: dict | None = None,
        prior2: dict | None = None,
    ) -> Answer:
        if bucket not in {"pairwise", "membership"}:
            return Answer(
                question=question, status=ToolStatus.UNAVAILABLE,
                summary=f"Unsupported bucket {bucket} (only pairwise/membership are supported).",
                bucket=bucket, task_type=task_type, candidates=list(candidates),
                abstained=True, confidence=0.0, rationale="out_of_scope",
            )
        mode = os.environ.get("SLC_DECISION_MODE", "auto").lower()

        # heuristic: pure heuristic (may abstain), used for the ablation contrast with baseline the baseline
        if mode == "heuristic":
            ans, skip_conf = await self._decide_fuse(
                self._heuristic(question, kg, ck, bucket, candidates, task_type, context),
                prior, question, kg, ck, bucket, candidates, task_type, context, prior2)
            return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))

        # auto / llm: no "direction first" short circuit; direction, magnitude and paralog buffering are all handed to the final agent (Vega LLM) as reference knowledge for synthesis
        ans = await self._llm_analogy(question, kg, ck, bucket, candidates, task_type, context or {})
        if ans is None:  # LLM unavailable -> falls back to the transparent heuristic (which already fuses direction and magnitude)
            h = self._heuristic(question, kg, ck, bucket, candidates, task_type, context)
            if h.abstained:
                h.rationale = "[LLM unavailable and the direct evidence is insufficient -> abstain honestly] " + h.rationale
            ans, skip_conf = await self._decide_fuse(h, prior, question, kg, ck,
                                              bucket, candidates, task_type, context, prior2)
            return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))
        ans, skip_conf = await self._decide_fuse(ans, prior, question, kg, ck,
                                          bucket, candidates, task_type, context, prior2)
        return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))

    # ------------------------------------------------------------------ fusion layer (prior + debate + LLM chief judge)

    # ------------------------------------------------------------------ answer-layer ensemble
    @staticmethod
    def _answer_ensemble_n() -> int:
        """Gate SLC_ANSWER_ENSEMBLE: how many fused decisions to draw (default 1, a single pass)."""
        try:
            return max(1, int(os.environ.get("SLC_ANSWER_ENSEMBLE", "1")))
        except (TypeError, ValueError):
            return 1

    async def _decide_fuse(self, ans: Answer, prior: dict | None, question: str,
                           kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                           candidates: list[str], task_type: str,
                           context: dict | None, prior2: dict | None = None) -> tuple[Answer, bool]:
        """Entry point of the fusion layer. When the triad-vote gate is on it wins; otherwise, when
        the ensemble size is above one and both the prior channel and the chief judge are enabled,
        the answer-layer ensemble runs; otherwise a single fusion pass decides.
        A supplied prior that is not None is the live signal that the prior channel is enabled, so
        it always matches the gate."""
        if self._triad_vote() and prior is not None:
            return await self._fuse_triad(ans, prior, question, kg, ck, bucket,
                                          candidates, task_type, context, prior2)
        if self._answer_ensemble_n() > 1 and prior is not None and self._llm_chief():
            return await self._fuse_ensemble(ans, question, kg, ck, bucket,
                                             candidates, task_type, context,
                                             self._answer_ensemble_n())
        return await self._fuse(ans, prior, question, kg, ck, bucket,
                                candidates, task_type, context)

    # ------------------------------------------------------------------ triad vote

    @staticmethod
    def _triad_vote() -> bool:
        """Gate SLC_TRIAD_VOTE (off by default, which keeps the single-fusion behaviour)."""
        return os.environ.get("SLC_TRIAD_VOTE", "0").lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _vote_key(bucket: str, winner, answer, abstained: bool) -> str | None:
        """Vote key of a channel: an abstention returns None and is not counted; pairwise votes look
        at winner, membership votes look at answer."""
        if abstained:
            return None
        if bucket == "pairwise":
            return f"W:{str(winner).upper()}" if winner else None
        return f"A:{bool(answer)}" if answer is not None else None

    @staticmethod
    def _triad_critique() -> bool:
        """Gate SLC_TRIAD_CRITIQUE (off by default, which keeps the plain triad vote)."""
        return os.environ.get("SLC_TRIAD_CRITIQUE", "0").lower() in {"1", "true", "on", "yes"}

    async def _fuse_triad(self, ans: Answer, prior: dict | None, question: str,
                          kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                          candidates: list[str], task_type: str,
                          context: dict | None, prior2: dict | None = None) -> tuple[Answer, bool]:
        """Triad vote: the evidence channel, a bare-prior LLM and a second prior draw with a
        different persona each cast one vote; when they disagree all three go to the final agent,
        which debates first and then decides.
        The second draw (supplied prior2, produced in parallel by the supply stage with a role-
        heterogeneous persona) is no longer taken serially, so its absence rate drops sharply. When
        SLC_TRIAD_CRITIQUE is on, a conflicted state first runs one round of parallel
        cross-critique: the two prior channels keep their personas, read each other's arguments,
        attack the weakest point and return a revised final vote, which decorrelates the votes
        inside the process itself. Only then do the three votes go to the final agent. The rest of
        the fallback chain is unchanged: a missing judge falls back to the deterministic majority;
        with no majority the prior outvotes the evidence channel; and if every channel abstains the
        honest abstention is kept."""
        from collections import Counter
        from knowledge_agents import prior_agent as _pa

        p1 = prior
        key_ev = self._vote_key(bucket, ans.winner, ans.answer, ans.abstained)
        key_p1 = self._vote_key(bucket, p1.get("winner"), p1.get("answer"),
                                bool(p1.get("abstained")))
        if key_ev is not None and key_ev == key_p1:
            # Two votes already agree. A third can only turn 2-0 into 2-1, and the majority is set,
            # so the answer passes through with zero extra calls.
            ans.rationale += (" |[triad vote: agreement] the evidence channel and the first pure-LLM "
                              "vote agree and the majority is set; no critique and no further vote "
                              "(zero extra cost).")
            return ans, False

        if prior2 is not None:
            p2 = prior2   # already drawn in parallel by the supply stage with a distinct persona
        else:
            t2 = float(os.environ.get("SLC_TRIAD_T2", "0.7"))
            p2 = await _pa.ask_prior_augmented(question, bucket, candidates,
                                               digest="", temperature=t2,
                                               persona=_pa.ROLE_PERSONAS["L2"]) or {}
        key_p2 = self._vote_key(bucket, p2.get("winner"), p2.get("answer"),
                                bool(p2.get("abstained"))) if p2 else None
        critiqued = False

        if self._triad_critique() and (key_p1 is not None or key_p2 is not None):
            ev_view = (f"winner={ans.winner} answer={ans.answer} conf={ans.confidence} "
                       f"rationale: {ans.rationale[:300]}") if not ans.abstained else \
                "abstained (insufficient graph/statistical evidence)"
            others_for_l1 = (f"L2(quant geneticist): {self._prior_view(p2) if key_p2 else 'abstained/absent'}\n"
                             f"E(evidence): {ev_view}")
            others_for_l2 = (f"L1(memory specialist): {self._prior_view(p1) if key_p1 else 'abstained/absent'}\n"
                             f"E(evidence): {ev_view}")
            c1, c2 = await asyncio.gather(
                _pa.critique_prior(question, bucket, candidates,
                                   _pa.ROLE_PERSONAS["L1"], self._prior_view(p1), others_for_l1),
                _pa.critique_prior(question, bucket, candidates,
                                   _pa.ROLE_PERSONAS["L2"],
                                   self._prior_view(p2) if key_p2 else "(no valid original vote)", others_for_l2),
                return_exceptions=True)
            critiqued = True
            if isinstance(c1, dict) and c1:
                p1 = c1
                key_p1 = self._vote_key(bucket, p1.get("winner"), p1.get("answer"),
                                        bool(p1.get("abstained")))
            if isinstance(c2, dict) and c2:
                p2 = c2
                key_p2 = self._vote_key(bucket, p2.get("winner"), p2.get("answer"),
                                        bool(p2.get("abstained")))

        votes = [("E(evidence channel)", key_ev,
                  (f"winner={ans.winner} answer={ans.answer} conf={ans.confidence} "
                   f"rationale: {ans.rationale[:300]}") if not ans.abstained
                  else "abstained (graph/statistical evidence insufficient)"),
                 ("L1(pure LLM, zero evidence)" + (" [critiqued]" if critiqued else ""),
                  key_p1, self._prior_view(p1)),
                 ("L2(pure LLM, independent draw)" + (" [critiqued]" if critiqued else ""),
                  key_p2,
                  self._prior_view(p2) if key_p2 else "abstained / call failed")]
        keys = [k for _, k, _ in votes if k is not None]
        cnt = Counter(keys)
        tally = dict(cnt)
        top_key, top_n = (cnt.most_common(1)[0] if cnt else (None, 0))
        majority = top_key if top_n >= 2 and list(cnt.values()).count(top_n) == 1 else None
        views = [(name, view if view else "no vote") for name, k, view in votes]
        facts = self._facts_digest(kg, ck, ans)

        j = await _pa.adjudicate_multi(question, bucket, candidates, facts, views,
                                       tally=str(tally))
        tag = "+crit" if critiqued else ""
        if j is not None:
            out = self._judge_answer(ans, j)
            out.basis = sorted(set(out.basis) | {"triad_vote"})
            out.decision_mode = f"{ans.decision_mode}+triad{tag}"
            lead = ("[triad vote + cross-critique]" if critiqued else "[triad vote]")
            out.rationale = (f"{lead} tally {tally}, the final agent decided after debate:") + \
                out.rationale.split("the evidence channel and the memory prior conflict, the judge re-examined independently ", 1)[-1]
            return out, True

        # judge absent/abstained -> fall back to the deterministic majority vote
        def _adopt_by_key(match_key: str) -> Answer | None:
            if match_key == key_ev and key_ev is not None:
                ans.rationale += f" |[triad vote] judge absent, the majority vote {match_key} agrees with the evidence channel and is adopted. Tally {tally}."
                return ans
            for p, k in ((p1, key_p1), (p2, key_p2)):
                if k == match_key and p:
                    fb = self._prior_fallback(ans, p, src=f"pure-LLM independent vote {k}")
                    if fb is not None:
                        fb.basis = sorted(set(fb.basis) | {"triad_vote"})
                        return fb
            return None

        if majority is not None:
            out = _adopt_by_key(majority)
            if out is not None:
                return out, (out is not ans)
        # No majority is impossible in the ordinary sense: the answer space is binary, so what is
        # left is a single vote or total abstention. The prior therefore beats the evidence channel.
        if key_p1 is not None:
            fb = self._prior_fallback(ans, p1)
            if fb is not None:
                fb.rationale = f"[triad vote: no majority, falling back to the prior rule; tally {tally}] " + fb.rationale
                return fb, True
        if key_p2 is not None:
            fb = self._prior_fallback(ans, p2, src="pure-LLM second vote")
            if fb is not None:
                return fb, True
        ans.rationale += f" |[triad vote] judge absent and no majority (tally {tally}); the evidence channel keeps its original verdict."
        return ans, False


    async def _fuse_ensemble(self, ans: Answer, question: str,
                             kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                             candidates: list[str], task_type: str,
                             context: dict | None, n: int) -> tuple[Answer, bool]:
        """Run the complete fused decision independently n times, drawing a fresh temperature-
        perturbed memory prior each time and letting each draw go through its own agree / judge /
        abstain branch, then take the mode over the final answer layer - the aggregation point is
        the decision point. If consensus falls below SLC_ENS_MIN_CONSENSUS the system abstains
        honestly instead of inventing an answer, restoring the abstention protection that a single
        fusion pass does not have."""
        import copy
        from knowledge_agents import prior_agent as _pa
        temps = ([round(0.2 + 0.6 * i / (n - 1), 2) for i in range(n)]
                 if n > 1 else [0.2])

        async def one(t: float):
            a = copy.deepcopy(ans)
            p = await _pa.ask_prior_augmented(question, bucket, candidates,
                                              digest="", temperature=t)
            return await self._fuse(a, p, question, kg, ck, bucket,
                                    candidates, task_type, context)

        raw = await asyncio.gather(*[one(t) for t in temps], return_exceptions=True)
        results = [r for r in raw if isinstance(r, tuple) and isinstance(r[0], Answer)]
        if not results:
            return ans, False
        return self._plurality(ans, results, bucket, n)

    @staticmethod
    def _plurality(base: Answer, results: list, bucket: str, n: int) -> tuple[Answer, bool]:
        """Vote over the n fused answers at the answer layer, counting abstentions separately and taking the mode of winner/answer as the result.
        If the top consensus share < the -> abstain honestly; otherwise adopt the modal representative decision and keep its provenance."""
        import copy
        from collections import Counter

        def dec_key(a: Answer) -> str:
            if a.abstained:
                return "__ABSTAIN__"
            return (("W:" + str(a.winner).upper()) if bucket == "pairwise"
                    else ("A:" + str(bool(a.answer))))

        keys = [dec_key(a) for a, _ in results]
        cnt = Counter(keys)
        top_key, top_n = cnt.most_common(1)[0]
        frac = top_n / len(results)
        tally = dict(cnt)
        min_frac = float(os.environ.get("SLC_ENS_MIN_CONSENSUS", "0.5"))
        rep_ans, rep_skip = next(r for r in results if dec_key(r[0]) == top_key)
        new = copy.deepcopy(rep_ans)
        new.decision_mode = f"{rep_ans.decision_mode}+ansens"
        new.basis = sorted(set(rep_ans.basis) | {"answer_ensemble"})
        if top_key == "__ABSTAIN__" or frac < min_frac:
            new.abstained = True
            new.winner = None
            new.answer = None
            new.confidence = round(frac * float(rep_ans.confidence or 0.0), 3)
            new.rationale = (f"[answer-layer ensemble: low-consensus abstention] {n} independent fused decisions voted {tally},"
                             f"the top consensus was only {top_key}={frac:.2f} < the threshold {min_frac}, so it abstains honestly (nothing invented)."
                             f"Representative decision provenance: {rep_ans.rationale[:300]}")
            return new, True
        new.rationale = (f"[answer-layer ensemble] {n} independent fused decisions voted {tally},"
                         f"mode {top_key} is adopted (consensus {frac:.2f}). Representative provenance: {rep_ans.rationale[:400]}")
        return new, rep_skip

    async def _fuse(self, ans: Answer, prior: dict | None, question: str,
                    kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                    candidates: list[str], task_type: str,
                    context: dict | None) -> tuple[Answer, bool]:
        """Return (the answer, whether conformal filtering was skipped). Only answers produced by
        the prior, the debate or the judge may skip it: the conformal thresholds were calibrated on
        the confidence semantics of the evidence channel, so applying them to a prior or a vote
        carries no statistical guarantee.
        When the SLC_PRIOR_AGENT gate is off, supplied['prior'] is None and this function is a
        pass-through. When SLC_LLM_CHIEF is set together with the prior gate, the design inverts the
        old fallback: diagnostics showed that the evidence channel answering on its own was right
        only about 33% of the time and that the bare prior was close to a coin flip, so abstention
        now goes to an evidence-augmented LLM and a conflict goes to an LLM judge. The factual data
        (graph and statistics) is injected as review material rather than used for scoring. With
        every gate off, this reproduces the plain prior-fusion behaviour."""
        if prior is None:
            return ans, False
        agree = ((bucket == "pairwise" and ans.winner and prior.get("winner")
                  and str(ans.winner).upper() == str(prior["winner"]).upper())) or \
                ((bucket == "membership" and ans.answer is not None
                  and prior.get("answer") is not None and bool(ans.answer) == bool(prior["answer"])))
        if not ans.abstained and agree:
            # Two independent channels agreeing is the strongest signal in the whole system: in
            # diagnostics this state was right about 77% of the time.
            ens = (f"(ensemble-prior agreement rate={prior.get('agreement')}, "
                   f"{prior.get('n_valid')}/{prior.get('n_draws')} valid votes)"
                   if "agreement" in prior else "")
            ans.rationale += f" |[prior channel: agreement] the LLM memory prior and the evidence channel agree at zero extra cost.{ens}"
            return ans, False

        from knowledge_agents import prior_agent as _pa
        if self._llm_chief():
            facts = self._facts_digest(kg, ck, ans)
            if ans.abstained:
                # the evidence channel abstained -> let a fact-bearing LLM re-answer (instead of a coin-flip bare prior)
                aug = await _pa.ask_prior_augmented(question, bucket, candidates, facts)
                if aug is not None and not aug.get("abstained"):
                    fb = self._prior_fallback(ans, aug, src="evidence-augmented LLM re-answer")
                    if fb is not None:
                        return fb, True
                fb = self._prior_fallback(ans, prior)   # If the augmented abstention/fails -> use the plain prior
                if fb is not None:
                    return fb, True
                return ans, False
            # answered but in conflict -> a single LLM judge reads both sides + the factual review material
            j = await _pa.adjudicate(question, bucket, candidates,
                                     facts, self._prior_view(prior))
            if j is not None:
                return self._judge_answer(ans, j), True
            # judge absent/refused to rule -> prior takes priority (the evidence channel's standalone accuracy is ~33%, below the prior)
            fb = self._prior_fallback(ans, prior)
            if fb is not None:
                return fb, True
            return ans, False

        from knowledge_agents import debate as _db

        if ans.abstained:
            # the evidence channel abstained -> so debate first (the prior is only a viewpoint under review) and fall back to the prior fallback
            if _db.debate_enabled():
                d = await self._debate_wrap(ans, prior, question, kg, ck, bucket,
                                            candidates, task_type, context)
                if d is not None:
                    return d, True
            fb = self._prior_fallback(ans, prior)
            if fb is not None:
                return fb, True
            return ans, False

        # Answered, but the answer and the prior disagree -> debate arbitration runs (it changes the answer only when enabled and decisive; otherwise the evidence channel keeps priority)
        if _db.debate_enabled():
            d = await self._debate_wrap(ans, prior, question, kg, ck, bucket,
                                         candidates, task_type, context)
            if d is not None:
                return d, True
        ans.rationale += (" |[prior channel: conflict not arbitrated] the memory prior "
                          f"{prior.get('winner') or prior.get('answer')}(conf={prior.get('confidence')})"
                          " disagrees with the evidence channel, the debate reached no verdict, so the evidence channel wins (a prior never overturns evidence).")
        return ans, False

    # ------------------------------------------------------------------ chief-judge components
    @staticmethod
    def _llm_chief() -> bool:
        return os.environ.get("SLC_LLM_CHIEF", "0").lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _facts_digest(kg: BackgroundKnowledge, ck: ComputedKnowledge,
                      ans: Answer) -> str:
        """The factual review material given to the augmented LLM/judge contains only observed values and graph relations, never the pipeline conclusion
        (conclusions anchor the LLM decision, and thelit-rag lesson of raw-passage pile-up degrading it => a structured summary + bounded length)."""
        import json as _j
        facts: dict = {}
        try:
            evs = []
            for ev in (kg.evidence or []):
                if ev.status == ToolStatus.OK and ev.rows:
                    evs.append({"op": ev.operation, "rows": ev.rows[:5]})
            if evs:
                facts["kg_observations"] = evs
            gc = ck.graph_computation
            if gc is not None and gc.status == ToolStatus.OK:
                ppr = sorted(({"gene": p.gene,
                               "ppr_degree_corrected": p.degree_corrected_score,
                               "in_graph": p.in_graph}
                              for p in gc.candidates if p.in_graph),
                             key=lambda x: -(x["ppr_degree_corrected"] or 0))[:4]
                if ppr:
                    facts["ppr"] = ppr
            ebg = ck.expression_by_gene or {}
            ex = {g: {"row_count": o.row_count} for g, o in ebg.items() if o.status == ToolStatus.OK}
            if ex:
                facts["timer_expr"] = ex
            sbg = ck.survival_by_gene or {}
            km = {g: (o.stats_text or "")[:100] for g, o in sbg.items() if o.status == ToolStatus.OK}
            if km:
                facts["km"] = km
            if ck.paralog_redundancy:
                facts["paralog_redundancy"] = ck.paralog_redundancy
            if ans.gene_scores:
                facts["pipeline_gene_scores"] = ans.gene_scores
        except Exception:  # noqa BLE001 - a failure to build the review material yields => empty facts (the augmentedLLMfalls back to the bare question)
            facts = {}
        s = _j.dumps(facts, ensure_ascii=False, default=str)
        return s[:1800]

    @staticmethod
    def _prior_view(prior: dict) -> str:
        base = (f"winner={prior.get('winner')} answer={prior.get('answer')} "
                f"conf={prior.get('confidence')} rationale: {str(prior.get('rationale', ''))[:300]}")
        if "agreement" in prior:   # ensemble prior: pass the agreement rate and vote pattern to the judge as reliability signals
            base += (f" | ensemble prior: {prior.get('n_valid')}/{prior.get('n_draws')} valid votes, "
                     f"agreement={prior.get('agreement')}, vote pattern={prior.get('votes')}")
        return base

    @staticmethod
    def _judge_answer(ans: Answer, j: dict) -> Answer:
        """The judge verdict generates a new answer: judge=LLM and its conf semantics are not the evidence channel's -> skip conformal filtering."""
        import copy
        new = copy.deepcopy(ans)
        new.abstained = False
        new.status = ToolStatus.OK
        new.winner = j.get("winner") if ans.bucket == "pairwise" else None
        new.answer = j.get("answer") if ans.bucket == "membership" else None
        new.confidence = round(float(j.get("confidence", 0.0) or 0.0), 3)
        new.basis = sorted(set(ans.basis) | {"llm_judge"})
        new.decision_mode = f"{ans.decision_mode}+judge"
        new.rationale = (f"[LLM chief judge: conflict ruling] the evidence channel and the memory prior conflict, the judge re-examined independently "
                         f"winner={new.winner} answer={new.answer} conf={new.confidence};"
                         f"Judge rationale: {str(j.get('rationale', ''))[:240]}."
                         f"Original evidence-channel decision: winner={ans.winner} answer={ans.answer} "
                         f"conf={ans.confidence}(kept for provenance).")
        return new

    @staticmethod
    def _prior_fallback(ans: Answer, prior: dict, src: str = "LLM memory prior") -> Answer | None:
        """When the evidence channel abstains, adopt the prior answer - label the source honestly and
        never pass it off as graph or statistical evidence. src is either "LLM memory prior" for a
        bare prior draw or "evidence-augmented LLM re-answer" for a re-answer built on injected
        facts."""
        import copy
        if prior.get("winner") is None and prior.get("answer") is None:
            return None
        fb = copy.deepcopy(ans)
        fb.abstained = False
        fb.status = ToolStatus.OK
        fb.winner = prior.get("winner")
        fb.answer = prior.get("answer")
        fb.confidence = round(float(prior.get("confidence", 0.0) or 0.0), 3)
        fb.basis = sorted(set(ans.basis) | {"prior_agent"})
        fb.decision_mode = f"{ans.decision_mode}+prior"
        ens = (f"(ensemble-prior agreement rate={prior.get('agreement')}, "
               f"{prior.get('n_valid')}/{prior.get('n_draws')} valid votes, vote pattern={prior.get('votes')})"
               if "agreement" in prior else "")
        fb.rationale = (f"[prior agent fallback] the evidence channel abstained (its text follows), adopting the {src} "
                        f"winner={fb.winner} answer={fb.answer} conf={fb.confidence}{ens};"
                        f"Prior rationale: {str(prior.get('rationale', ''))[:200]}."
                        f"Note: this answer is not graph/statistical evidence but a memory prior. || Original abstention rationale: {ans.rationale[:300]}")
        return fb

    async def _debate_wrap(self, ans: Answer, prior: dict, question: str,
                           kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                           candidates: list[str], task_type: str,
                           context: dict | None) -> Answer | None:
        """Run one round of 3 expert debate and return None if no verdict emerges."""
        from knowledge_agents import debate as _db
        digest = (f"Evidence-channel decision: winner={ans.winner} answer={ans.answer} "
                  f"abstained={ans.abstained} conf={ans.confidence} "
                  f"gene_scores={ans.gene_scores}\nDecision rationale: {ans.rationale[:600]}")
        views = (f"prior(pureLLMmemory prior): winner={prior.get('winner')} answer={prior.get('answer')} "
                 f"conf={prior.get('confidence')} rationale: {str(prior.get('rationale', ''))[:300]}")
        d = await _db.run_debate(question, bucket, candidates, digest, views)
        if d is None:
            return None
        # If the debate verdict agrees with the original decision, append only a confirmation note;
        # otherwise build a new answer, whose voted confidence semantics bypass conformal filtering.
        same = ((bucket == "pairwise" and ans.winner and str(d.get("winner", '')).upper()
                 == str(ans.winner).upper()) or
                (bucket == "membership" and ans.answer is not None
                 and d.get("answer") is not None and bool(ans.answer) == bool(d["answer"])))
        if same and not ans.abstained:
            ans.rationale += f" |[expert debate: confirmed] {d['rationale']} {d['support']}."
            return ans
        import copy
        new = copy.deepcopy(ans)
        new.abstained = False
        new.status = ToolStatus.OK
        new.winner = d.get("winner") if bucket == "pairwise" else None
        new.answer = d.get("answer") if bucket == "membership" else None
        new.confidence = round(max(0.0, min(1.0, 0.5 + float(d.get("margin", 0.0)))), 3)
        new.basis = sorted(set(ans.basis) | {"expert_debate"})
        new.decision_mode = f"{ans.decision_mode}+debate"
        new.rationale = (f"{d['rationale']} {d['support']} tally={d['tally']};"
                         f"Original evidence-channel decision: winner={ans.winner} answer={ans.answer} "
                         f"abstained={ans.abstained}(conf={ans.confidence});"
                         f"prior view: winner={prior.get('winner')} answer={prior.get('answer')}"
                         f"(conf={prior.get('confidence')}). The voted conf does not go through conformal filtering (the threshold semantics do not apply).")
        return new

    # ------------------------------------------------------------------ Conformal abstention (calibrated abstention)
    @classmethod
    def _conformal_thresholds(cls) -> dict:
        """Read the JSON pointed to by SLC_CONFORMAL_FILE as {bucket: threshold}; it is produced by
        the conformal calibration step. When the gate is off, or the file is missing, an empty dict
        is returned so nothing is applied and the original decision stands. The cache lives as a
        module-level global because it cannot be a class attribute of a pydantic Action model (it
        would turn into a ModelPrivateAttr)."""
        global _CONFORMAL_CACHE
        if _CONFORMAL_CACHE is not None:
            return _CONFORMAL_CACHE
        path = os.environ.get("SLC_CONFORMAL_FILE", "").strip()
        data: dict = {}
        if path and os.path.exists(path):
            try:
                raw = json.loads(open(path, encoding="utf-8").read())
                data = {str(k).lower(): float(v) for k, v in (raw.get("thresholds") or raw).items()
                        if isinstance(v, (int, float))}
            except Exception:  # noqa BLE001 - a corrupt threshold file is treated as disabled and does not disturb the main flow
                data = {}
        _CONFORMAL_CACHE = data
        return data

    # ------------------------------------------------------------------ selective abstention (a risk-control knob)
    @staticmethod
    def _abstain_theta() -> float:
        """Gate SLC_ABSTAIN_THETA (0 by default, which leaves the frozen behaviour untouched)."""
        try:
            return max(0.0, min(1.0, float(os.environ.get("SLC_ABSTAIN_THETA", "0") or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def _maybe_selective_abstain(self, ans: Answer) -> Answer:
        """Turn a final answer whose confidence is below theta into an honest abstention. This also
        covers the triad-vote and judge paths. The semantics match conformal abstention but the
        source differs: the conformal threshold comes from a calibration set that bounds the error
        rate of answered cases, whereas theta is a risk operating point chosen by the user. The
        confidence distribution is written into the trace artifacts, so any recorded value can be
        re-simulated offline in a risk-coverage analysis without re-running the pipeline."""
        th = self._abstain_theta()
        if th <= 0.0 or ans.abstained:
            return ans
        if float(ans.confidence or 0.0) < th:
            ans.abstained = True
            ans.winner = None
            ans.answer = None
            ans.status = ToolStatus.UNAVAILABLE
            ans.rationale = f"[selective abstention conf={float(ans.confidence or 0.0):.2f} < theta={th:.2f}] " + ans.rationale
            ans.basis = sorted(set(ans.basis or []) | {"selective_abstain"})
        return ans

    @classmethod
    def _conformal_filter(cls, ans: Answer, bucket: str) -> Answer:
        """Conformal guarantee: when the confidence falls below the calibrated threshold, the answer
        is reclassified from answered to abstained - no fabrication and no reckless answering. The
        threshold comes from a calibration set disjoint from the test set and bounds the error rate
        of answered cases by alpha, which brings the over-confidence of the analogy path down to the
        chosen risk level."""
        thr = cls._conformal_thresholds().get((bucket or "").lower())
        if thr is None or ans.abstained:
            return ans
        ans.conformal_threshold = thr
        if ans.confidence < thr:
            ans.abstained = True
            ans.winner = None
            ans.answer = None
            ans.status = ToolStatus.UNAVAILABLE
            ans.rationale = (f"[conformal abstention conf={ans.confidence:.2f}<threshold{thr:.2f}] " + ans.rationale)
        return ans
