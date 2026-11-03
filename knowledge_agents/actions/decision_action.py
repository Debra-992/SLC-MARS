# -*- coding: utf-8 -*-
"""决策智能体核心动作：消费 BackgroundKnowledge + ComputedKnowledge，产出最终 Answer。

范围限 pairwise / membership。两种判定内核（env SLC_DECISION_MODE=heuristic|llm|auto）：
  - heuristic（默认，环境无关、确定性、便于评测）：对候选基因做透明加权打分
  - llm/auto：尝试用 MetaGPT LLM 综合证据出答案，超时/失败自动回退 heuristic

证据完整性不变量：候选无任何可用证据 -> 弃答（abstained），绝不把"未知"当阴性或乱选 winner。
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

# 各证据通道的加权（透明、可调；不是生物学先验，仅排序启发式）
W_KG_DISEASE_PATH = 0.5     # 每命中一条 基因-事件-疾病 路径
W_DIRECTION = 0.6           # 时间因果方向(rel_AB⊗rel_BC)与题目动词一致/相悖的加/减分
W_KG_NEIGHBOR = 0.05        # 每命中一条图谱邻居（家族/互作等，弱信号）
W_PPR = 1.0                 # 网络邻近度归一后（基因↔疾病关系主信号）
W_PPR_RANK = 0.3            # 基因进入疾病 PPR 候选榜的额外排名奖励
W_KM = 0.5                  # -log10(p) 截断后
W_DE = 0.25                 # |log2fc| 截断后
W_TIMER = 0.15             # TIMER 有可用表达证据（表或图）
W_BUFFER = 0.8             # 旁系缓冲：冗余∈[0,1] 对"必需/阳性"极性的单边削弱权重（与因果方向近平权）
MIN_SCORE_TO_ANSWER = 0.5   # 单基因证据总分低于此 -> 弃答
NORM_STOP = frozenset({"cancer", "cancers", "tumor", "tumour", "syndrome", "carcinoma", "disease", "diseases"})

# 保形阈值模块级缓存（None=未加载；{}=未启用/无文件）。不能作 pydantic Action 的类属性。
_CONFORMAL_CACHE: dict | None = None


def _sign_of(effect) -> int:
    """复合符号 gene_disease_effect -> +1 PROMOTION / -1 SUPPRESSION / 0 未知。"""
    e = str(effect or "").upper()
    return 1 if e == "PROMOTION" else (-1 if e == "SUPPRESSION" else 0)


# ---------------------------------------------------------------- pairwise 图特征 logistic（v10）
# 按解析出的工件路径缓存：env SLC_PAIRWISE_MODEL 未设 -> None（纯 v9 规则）；设了且文件在 -> 载入。
# 推理只应用已存系数（线性 + sigmoid），不依赖 sklearn，避免在线环境差异。
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
        except Exception:  # noqa BLE001 —— 工件坏即回退规则，绝不阻断判定
            model = None
    _PAIRWISE_MODEL_CACHE[path] = model
    return model


def _score_pairwise_graph(kg: BackgroundKnowledge, candidates: list[str],
                          cancer: str) -> tuple[list[str], dict[str, float], dict] | None:
    """用图特征 logistic 给每个候选基因打 P(是 winner) 分。
    返回 (按分降序的候选, {gene: prob}, 模型) —— 无工件 / 两候选皆无 gene_disease_paths 图证据
    (model 无从排序，须回退规则、不伪造) 时返回 None。"""
    model = _pairwise_model()
    if not model or len(candidates) < 2:
        return None
    rows = collect_path_rows(kg)
    feats = {g: gene_graph_feat(g, rows, cancer) for g in candidates}
    if all(feats[g].get("n_paths", 0) == 0 for g in candidates):
        return None                                   # 图谱链无任何证据 -> 诚实回退规则
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
    """决策动作：产出 Answer。"""

    name: str = "AnswerDecision"

    DECISION_PROMPT: str = """你在为 SLC 基因 CRISPR 筛选做证据综合判定。只依据下面提供的证据，不得臆造。
证据完整性：某通道为 unavailable/unknown/rejected 表示"未知"，绝不等同于阴性。
综合原则：因果方向(基因→疾病的 gene_disease_effect, 来自 RelaEvent rel_AB⊗rel_BC) 与 计算量级
(PPR 网络邻近度 / TIMER 表达 / KM 生存 / DESeq2 差异表达) 都是【参考知识】，需一起权衡、不得让任一通道独裁：
方向提供"促/抑"极性，量级提供"证据强弱/相关度"；二者冲突时如实反映在 confidence 上。
注意：疾病关联/邻近/差异表达 ≠ CRISPR loss-of-function 必需性，别把"高度关联"直接当成阳性。
旁系缓冲(paralog redundancy)：若候选基因在同癌种有大量同亚家族旁系共表达(redundancy 高)，
其 knockout 更易被旁系补偿 -> 更可能"非必需/阴性"；这是削弱阳性的单边证据，"无旁系"不等于必需。

题目(桶={bucket}): {question}
候选基因: {candidates}
题目约束: {context}

[背景知识 图谱+文献（含因果方向）]
{kg_json}

[计算知识 PPR/TIMER/KM/DE]
{ck_json}

请只输出一个 JSON：
- pairwise: {{"winner": "<候选之一或null>", "confidence": 0~1, "abstained": true/false, "rationale": "..."}}
- membership: {{"answer": true/false/null, "confidence": 0~1, "abstained": true/false, "rationale": "..."}}
证据不足时 winner/answer 置 null 且 abstained=true。
"""

    # 目标癌种证据不足时：把已有结果当知识，引导大模型推理『临近癌种 ↔ 目标癌种』关系后作答
    ANALOGY_PROMPT: str = """你在为 SLC 基因 CRISPR 筛选做判定。目标癌种【{target_cancer}】上，候选基因缺乏直接的图谱邻近/组学证据。
不要把"缺证据"当成阴性——请改用跨癌种的生物学推理。
把因果方向(gene_disease_effect)与计算量级(PPR/TIMER/KM/DE)都当【参考知识】一起权衡，任一通道都不独裁；关联/邻近/表达 ≠ CRISPR 必需性。

下面是该基因已有的全部证据（含它在【其它癌种/疾病】上的关联），把这些当知识：
[背景知识 图谱+文献]
{kg_brief}
[计算知识 PPR/TIMER/KM/DE]
{ck_brief}

该基因已知临近/关联的『临近癌种』: {related_cancers}
目标癌种: {target_cancer}   题目桶: {bucket}   候选基因: {candidates}
题目: {question}

请推理：这些临近癌种与目标癌种在组织来源、细胞谱系、分子分型、通路依赖上的相似/可迁移程度，
据此判断该(候选)基因在目标癌种中最可能的作用方向，并【必须给出一个明确答案】（不允许弃权）：
- pairwise: 从候选中二选一（选证据/类比上更可能的那一个）
- membership: true 或 false
只输出 JSON：
- pairwise: {{"winner": "<候选之一>", "confidence": 0~1, "abstained": false, "analogy": "<临近→目标的推理>", "rationale": "..."}}
- membership: {{"answer": true/false, "confidence": 0~1, "abstained": false, "analogy": "<临近→目标的推理>", "rationale": "..."}}
confidence 应反映跨癌种类推的不确定性（通常低于直接证据）。
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
        """返回 (strength, pol, buf, basis)：
          strength[g] = 各通道『量级/关联』总证据(无符号, ≥0)；
          pol[g]      = 时间因果方向的带符号净票 Σ(verb × sign)，只由 gene_disease_effect 提供方向；
          buf[g]      = 旁系缓冲极性 ∈ [-1, 0]（-冗余），冗余越高越把"必需/阳性"往阴性削；单边、无数据记 0；
          basis[g]    = 命中通道标签。
        方向与量级分开累计，避免把『无符号邻近度/表达』误当成极性证据。"""
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

        # 1) KG 疾病路径 / 邻居（来自 BackgroundKnowledge.evidence）
        for ev in kg.evidence:
            for row in ev.rows or []:
                if not isinstance(row, dict):
                    continue
                g = str(row.get("gene", "")).upper()
                if ev.operation == "gene_disease_paths":
                    bump(g, W_KG_DISEASE_PATH, "kg_disease_path")   # 关联强度（量级，无符号）
                    if verb:                                        # 因果方向（独立记入 pol，不改 strength）
                        s = _sign_of(row.get("gene_disease_effect"))
                        if s:
                            bump_pol(g, verb * s)
                            basis[g].add("kg_direction" if verb * s > 0 else "kg_direction_against")
                elif ev.operation == "gene_neighbors":
                    bump(g, W_KG_NEIGHBOR, "kg_neighbor")

        # 2) PPR 邻近度（归一到候选内的相对值 + 上榜排名奖励）
        if ck.graph_computation and ck.graph_computation.status is ToolStatus.OK:
            cands = ck.graph_computation.candidates or []
            vals = {c.gene.upper(): (c.degree_corrected_score or 0.0) for c in cands}
            mx = max(vals.values(), default=0.0) or 1.0
            n = max(1, len(cands))
            for idx, c in enumerate(cands):
                g = c.gene.upper()
                bump(g, W_PPR * ((c.degree_corrected_score or 0.0) / mx), "ppr")
                bump(g, W_PPR_RANK * (1.0 - idx / n), "ppr_rank")  # 越靠前越像疾病相关基因

        # 3) TIMER 表达证据（有表或有箱线图即计弱阳性）
        expr_map = ck.expression_by_gene or ({"_": ck.expression} if ck.expression else {})
        for gene, ex in expr_map.items():
            if gene == "_" or not ex or ex.status is not ToolStatus.OK:
                continue
            has_evidence = bool(ex.rows) or bool(ex.boxplot_path) or (ex.row_count or 0) > 0
            if has_evidence:
                bump(gene.upper(), W_TIMER, "timer")

        # 4) KM 生存显著性（逐基因）
        surv = ck.survival_by_gene or ({"_": ck.survival} if ck.survival else {})
        for gene, km in surv.items():
            if gene == "_" or not km or km.status is not ToolStatus.OK:
                continue
            bump(gene.upper(), W_KM * (cls._km_minuslog10(km.stats_text) / 10.0), "km")

        # 5) DESeq2 差异表达（癌种级，命中 top_genes 的候选才有）
        de = ck.differential_expression
        if de and de.status is ToolStatus.OK:
            for row in de.top_genes or []:
                g = str(row.get("gene_symbol") or row.get("gene") or "").upper()
                l2 = row.get("log2fc")
                if l2 is not None:
                    bump(g, W_DE * min(abs(float(l2)), 5.0) / 5.0, "de")

        # 6) 旁系缓冲（paralog redundancy -> 削弱必需性极性；单边、非负贡献给 strength）
        for gene_u, red in (ck.paralog_redundancy or {}).items():
            g = gene_u.upper()
            if g in buf:
                r = max(0.0, min(1.0, float(red or 0.0)))
                buf[g] = -r                      # ∈ [-1,0]
                if r > 0:
                    basis[g].add("paralog_buffered")
                    bump(g, 0.0, "paralog_redundancy")  # 记入 basis，不改 strength
                else:
                    basis[g].add("paralog_solo")

        return strength, pol, buf, basis

    # ------------------------------------------------------------------ 时间因果方向（rel_AB ⊗ rel_BC 复合符号）
    @classmethod
    def _norm_terms(cls, text: str) -> set[str]:
        return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
                if len(w) > 3 and w not in NORM_STOP}

    @staticmethod
    def _verb_dir(task_type: str) -> int:
        t = (task_type or "").lower()
        if t.endswith("positive"):
            return 1    # promote / 促生长：期望 gene→disease = PROMOTION
        if t.endswith("negative"):
            return -1   # inhibit / 抑制：期望 gene→disease = SUPPRESSION
        return 0

    @staticmethod
    def _direction(kg: BackgroundKnowledge, gene: str, cancer: str) -> dict:
        """统计该基因经 RelaEvent 复合符号 gene_disease_effect 的 促/抑 计数。
        优先题目癌种匹配的疾病行(matched=True)；癌种无匹配则回退全疾病(matched=False)。"""
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
        """确定性消融基线：不加『方向优先』短路，把因果方向(pol)与量级(strength)一并纳入
        综合证据后判定。方向只提供极性、量级只提供强度/排序，谁都不单独一票否决对方。"""
        cancer = (context or {}).get("cancer", "") or self._infer_cancer(question)
        verb = self._verb_dir(task_type)
        strength, pol, buf, basis = self._collect_scores(kg, ck, candidates, verb)

        if bucket == "pairwise":
            # v10：优先用 pairwise 图特征 logistic（env SLC_PAIRWISE_MODEL 指定工件时）；
            # 未启用、或两候选沿 SLC->Pathway->Disease 链均无证据（model 无从排序）时，
            # 诚实回退下面的 v9 透明加权规则，绝不凭空造 winner。
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
                                  rationale=f"[pairwise 图特征logistic·v10] {best}(P(winner)={probs[best]:.2f}) vs "
                                            f"{second}(P={probs.get(second, 0):.2f}), 差={gap:.2f}；"
                                            f"依据 SLC→Pathway→Disease 拆细图特征经学习式 logistic 排序"
                                            f"(留出准确率≈{ho_txt})。为关联方向代理，非 CRISPR 读数。")
            # 综合：量级 + 方向(带符号) + 旁系缓冲(冗余把候选往"更不必需"一侧压)
            combined = {g: strength[g] + W_DIRECTION * pol[g] + W_BUFFER * buf[g] for g in candidates}
            ranked = sorted(candidates, key=lambda g: (-combined[g], g))
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            if strength[best] < MIN_SCORE_TO_ANSWER and pol[best] == 0 and buf[best] == 0:
                return self._make(bucket, task_type, candidates, strength, basis,
                                  winner=None, answer=None, abstained=True, confidence=0.0,
                                  rationale=f"方向/量级/旁系缓冲证据均不足(best={best} strength={strength[best]:.2f}, pol=0, buf=0)，弃答不伪造。")
            gap = combined[best] - (combined[second] if second else combined[best])
            conf = max(0.0, min(1.0, 0.5 * math.tanh(gap) + 0.1 * min(1.0, strength[best])))
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=best, answer=None, abstained=False, confidence=round(conf, 3),
                              rationale=f"[方向+量级+旁系综合] best={best}(combined={combined[best]:.2f} pol={pol[best]:+.0f} "
                                        f"buf={buf[best]:+.2f} strength={strength[best]:.2f}) vs {second}(combined={combined.get(second, 0):.2f})，"
                                        f"证据: {sorted(basis[best])}。")

        # membership
        g = candidates[0] if candidates else ""
        p = pol.get(g, 0.0)
        b = buf.get(g, 0.0)
        s = strength.get(g, 0.0)
        # R2（env SLC_MEM_R2=1）：membership 改走“必需性(loss-of-function)证据重排”，去毒自相矛盾的促/抑极性。
        # 默认关 => 下方 v16 逻辑逐字节不变。
        if os.environ.get("SLC_MEM_R2", "0") == "1":
            return self._membership_r2(bucket, task_type, candidates, strength, basis, g, s, verb)
        # ① 关键修正：membership 的"阴/阳"极性只由因果方向 p 决定；旁系缓冲只能【削弱】
        # 已有方向的净强度，绝不能在"无因果方向"时单独把基因臆断为阴性（unknown≠negative）。
        # 旧逻辑 net=p+W_BUFFER*b 在 p=0 时仅凭缓冲(net=0.8*b<0)就大量误判阴性 -> 制造假阴。
        if p == 0:
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=None, answer=None, abstained=True, confidence=0.0,
                              rationale=f"{g} 仅有量级/关联证据(strength={s:.2f}, 通道{sorted(basis.get(g, set()))})、"
                                        f"无因果极性方向(rela 促/抑未确立)，旁系缓冲不足以单独定阴性，保持未知、诚实弃答。")
        net = p + W_BUFFER * b            # 有方向时：缓冲只削弱该方向的净强度（单边，主要削阳性）
        if abs(net) < 1e-9:
            # 缓冲把净极性恰好抵消到 0 -> 无法定向，诚实弃答
            return self._make(bucket, task_type, candidates, strength, basis,
                              winner=None, answer=None, abstained=True, confidence=0.0,
                              rationale=f"{g} 因果方向被旁系缓冲恰好抵消(net=0)，不足以判定，保持未知。")
        ans = net > 0
        conf = max(0.0, min(1.0, 0.3 + 0.12 * abs(net) + 0.1 * math.log1p(s)))
        return self._make(bucket, task_type, candidates, strength, basis,
                          winner=None, answer=ans, abstained=False, confidence=round(conf, 3),
                          rationale=f"[方向定极性+缓冲削弱] {g} 净极性 net={net:+.2f}"
                                    f"(因果 pol={p:+.0f}、旁系缓冲 buf={b:+.2f}×{W_BUFFER}；动词{'promote' if verb > 0 else 'inhibit' if verb < 0 else '?'})、"
                                    f"量级 strength={s:.2f} 通道{sorted(basis.get(g, set()))} -> {'阳性' if ans else '阴性'}"
                                    f"（极性仅由因果方向决定、缓冲只削弱、量级定置信，均 KG/组学证据，非 CRISPR 读数）。")

    def _membership_r2(self, bucket, task_type, candidates, strength, basis, g, s, verb) -> Answer:
        """R2：把 membership 从“极性方向定阴阳”改为“必需性(loss-of-function)证据重排”。
        自相矛盾的促/抑文献极性不再当作定阴阳的证据；主判据=极性矛盾(→必需)、旁系孤儿(→必需)、
        旁系缓冲(→非必需)、量级(→必需)。系数由 v16@n=53 membership 5-fold CV 拟合(样本外 acc 66.8%>地板60.4%)。
        只改 membership 判定、不弃权（利于 acc 全体）；pairwise / gate-off 路径完全不受影响。"""
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
                          rationale=f"[R2 必需性重排·v20] {g} 极性矛盾={conflict:.0f} 方向证据={dir_present:.0f} "
                                    f"旁系孤儿={solo:.0f} 缓冲冗余={buffered:.0f} 量级={s:.2f} -> "
                                    f"P(必需)={prob:.3f} -> {'必需' if ess else '非必需'}"
                                    f"（construct=CRISPR loss-of-function 必需性，非促/抑方向；"
                                    f"动词{'promote' if verb > 0 else 'inhibit' if verb < 0 else '?'}）。")

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
        except Exception:  # noqa BLE001 —— LLM 不可用即回退，不阻断管线
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
            rationale = f"[跨癌种类推] {data['analogy']} | {rationale}"
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

    # ------------------------------------------------------------------ 跨癌种类推
    @staticmethod
    def _related_cancers(kg: BackgroundKnowledge, candidates: list[str]) -> list[str]:
        """从 KG gene_disease_paths 提取候选基因已知关联的『临近癌种/疾病』。"""
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
        """把 kg/ck 压成简短文本喂给 LLM（只留有内容的回执，控长度）。"""
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
                lines.append(f"- PPR疾病邻近Top: {top}")
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
                    lines.append(f"- 旁系冗余[{g}]: redundancy={r:.2f} 同亚家族共表达旁系={ (pmap.get(g) or [])[:6] }"
                                 f"（冗余高=knockout 易被缓冲=更可能非必需/阴性）")
        return "\n".join(lines) or "(无明显证据)"

    async def _llm_analogy(self, question, kg, ck, bucket, candidates, task_type, context):
        target_cancer = (context or {}).get("cancer", "") or self._infer_cancer(question)
        related = self._related_cancers(kg, candidates)
        prompt = self.ANALOGY_PROMPT.format(
            target_cancer=target_cancer or "未指定",
            kg_brief=self._evidence_brief(kg), ck_brief=self._evidence_brief(ck),
            related_cancers=related or "（KG 未给出关联癌种，请基于癌种生物学常识推理）",
            bucket=bucket, candidates=candidates, question=question,
        )
        try:
            rsp = await asyncio.wait_for(self._aask(prompt),
                                         timeout=float(os.environ.get("SLC_DECISION_LLM_TIMEOUT", "40")))
        except Exception:  # noqa BLE001 —— LLM 不可用返回 None，由 run() 兜底
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
                summary=f"暂不支持的桶 {bucket}（当前限 pairwise/membership）。",
                bucket=bucket, task_type=task_type, candidates=list(candidates),
                abstained=True, confidence=0.0, rationale="out_of_scope",
            )
        mode = os.environ.get("SLC_DECISION_MODE", "auto").lower()

        # heuristic：纯启发式（可弃答），用于与 baseline 的消融对照
        if mode == "heuristic":
            ans, skip_conf = await self._decide_fuse(
                self._heuristic(question, kg, ck, bucket, candidates, task_type, context),
                prior, question, kg, ck, bucket, candidates, task_type, context, prior2)
            return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))

        # auto / llm：不做"方向优先"短路，把方向+量级+旁系缓冲全部作为参考知识交给最终智能体(Vega LLM)综合判定
        ans = await self._llm_analogy(question, kg, ck, bucket, candidates, task_type, context or {})
        if ans is None:  # LLM 不可用 -> 回退到透明启发式（其内部已综合方向与量级）
            h = self._heuristic(question, kg, ck, bucket, candidates, task_type, context)
            if h.abstained:
                h.rationale = "[LLM 不可用且直接证据不足 -> 诚实弃答] " + h.rationale
            ans, skip_conf = await self._decide_fuse(h, prior, question, kg, ck,
                                              bucket, candidates, task_type, context, prior2)
            return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))
        ans, skip_conf = await self._decide_fuse(ans, prior, question, kg, ck,
                                          bucket, candidates, task_type, context, prior2)
        return self._maybe_selective_abstain(ans if skip_conf else self._conformal_filter(ans, bucket))

    # ------------------------------------------------------------------ v11/v12 融合层（prior + 辩论 + LLM 主审）

    # ------------------------------------------------------------------ v14 答案层集成
    @staticmethod
    def _answer_ensemble_n() -> int:
        """v14 门控：SLC_ANSWER_ENSEMBLE（默认 1 = 单次融合 = v12 逐字节）。"""
        try:
            return max(1, int(os.environ.get("SLC_ANSWER_ENSEMBLE", "1")))
        except (TypeError, ValueError):
            return 1

    async def _decide_fuse(self, ans: Answer, prior: dict | None, question: str,
                           kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                           candidates: list[str], task_type: str,
                           context: dict | None, prior2: dict | None = None) -> tuple[Answer, bool]:
        """融合层入口。v15/v16 三通道表决门控优先；否则 N>1 且 prior 通道开启且主审开启
        -> v14 答案层集成；否则单次 _fuse(v12)。
        用 supplied prior 非 None 作为"先验通道已启用"的活动信号（与门控一致）。"""
        if self._triad_vote() and prior is not None:
            return await self._fuse_triad(ans, prior, question, kg, ck, bucket,
                                          candidates, task_type, context, prior2)
        if self._answer_ensemble_n() > 1 and prior is not None and self._llm_chief():
            return await self._fuse_ensemble(ans, question, kg, ck, bucket,
                                             candidates, task_type, context,
                                             self._answer_ensemble_n())
        return await self._fuse(ans, prior, question, kg, ck, bucket,
                                candidates, task_type, context)

    # ------------------------------------------------------------------ v15 三通道表决

    @staticmethod
    def _triad_vote() -> bool:
        """v15 门控：SLC_TRIAD_VOTE（默认关 = v12 行为逐字节不变）。"""
        return os.environ.get("SLC_TRIAD_VOTE", "0").lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _vote_key(bucket: str, winner, answer, abstained: bool) -> str | None:
        """通道的表决键：弃权=None（不计票）；pairwise 看 winner；membership 看 answer。"""
        if abstained:
            return None
        if bucket == "pairwise":
            return f"W:{str(winner).upper()}" if winner else None
        return f"A:{bool(answer)}" if answer is not None else None

    @staticmethod
    def _triad_critique() -> bool:
        """v16 门控：SLC_TRIAD_CRITIQUE（默认关 = v15 行为逐字节不变）。"""
        return os.environ.get("SLC_TRIAD_CRITIQUE", "0").lower() in {"1", "true", "on", "yes"}

    async def _fuse_triad(self, ans: Answer, prior: dict | None, question: str,
                          kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                          candidates: list[str], task_type: str,
                          context: dict | None, prior2: dict | None = None) -> tuple[Answer, bool]:
        """v15：证据通道 ∥ 纯LLM（零证据）×2 —— 冲突时三票呈最终智能体先辩论后表决。
        v16 升级（需 supplied prior2，由 supply 段并行抽出，角色异构 persona）：
        L2 不再串行现抽（缺席率大降）；SLC_TRIAD_CRITIQUE 开启时冲突态先走一轮
        【并行交叉批评】（L1/L2 保持各 persona，互看论点攻最弱后给修订终票，
        mdagents 式过程内去相关），再三票呈最终智能体表决。其余退链不变：
        法官缺席→确定性多数；无多数→v12 规则 prior 胜证据；全弃权→维持诚实弃答。"""
        from collections import Counter
        from knowledge_agents import prior_agent as _pa

        p1 = prior
        key_ev = self._vote_key(bucket, ans.winner, ans.answer, ans.abstained)
        key_p1 = self._vote_key(bucket, p1.get("winner"), p1.get("answer"),
                                bool(p1.get("abstained")))
        if key_ev is not None and key_ev == key_p1:
            # 两票同向：第三票最多把 2-0 变 2-1，多数已定 -> 零额外调用直通（同 v12 agree）
            ans.rationale += (" ｜[三通道·v15 一致] 证据通道与纯LLM票1同向，多数已定，"
                              "不再走批评/表决（零额外成本）。")
            return ans, False

        if prior2 is not None:
            p2 = prior2                      # v16：supply 段已并行抽好（含角色 persona/竞速）
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
                       f"理由: {ans.rationale[:300]}") if not ans.abstained else \
                "弃权（图谱/统计证据不足）"
            others_for_l1 = (f"L2(quant geneticist): {self._prior_view(p2) if key_p2 else '弃权/缺席'}\n"
                             f"E(evidence): {ev_view}")
            others_for_l2 = (f"L1(memory specialist): {self._prior_view(p1) if key_p1 else '弃权/缺席'}\n"
                             f"E(evidence): {ev_view}")
            c1, c2 = await asyncio.gather(
                _pa.critique_prior(question, bucket, candidates,
                                   _pa.ROLE_PERSONAS["L1"], self._prior_view(p1), others_for_l1),
                _pa.critique_prior(question, bucket, candidates,
                                   _pa.ROLE_PERSONAS["L2"],
                                   self._prior_view(p2) if key_p2 else "(无有效原票)", others_for_l2),
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
                   f"理由: {ans.rationale[:300]}") if not ans.abstained
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
        tag = "+crit16" if critiqued else ""
        if j is not None:
            out = self._judge_answer(ans, j)
            out.basis = sorted(set(out.basis) | {"triad_vote"})
            out.decision_mode = f"{ans.decision_mode}+triad{tag}"
            lead = "[三通道表决+交叉批评·v16]" if critiqued else "[三通道表决·v15]"
            out.rationale = (f"{lead} 票型 {tally}，最终智能体辩论后裁决：") + \
                out.rationale.split("，法官独立复核后 ", 1)[-1]
            return out, True

        # 法官缺席/弃权 -> 确定性多数票退回
        def _adopt_by_key(match_key: str) -> Answer | None:
            if match_key == key_ev and key_ev is not None:
                ans.rationale += f" ｜[三通道·v15] 法官缺席，多数票 {match_key} 与证据通道一致，采纳。票型 {tally}。"
                return ans
            for p, k in ((p1, key_p1), (p2, key_p2)):
                if k == match_key and p:
                    fb = self._prior_fallback(ans, p, src=f"纯LLM独立票·v15({k})")
                    if fb is not None:
                        fb.basis = sorted(set(fb.basis) | {"triad_vote"})
                        return fb
            return None

        if majority is not None:
            out = _adopt_by_key(majority)
            if out is not None:
                return out, (out is not ans)
        # 无多数（1-1-1 不可能：答案空间二值；仅剩单票或全弃权）-> v12 规则：prior 胜过证据通道
        if key_p1 is not None:
            fb = self._prior_fallback(ans, p1)
            if fb is not None:
                fb.rationale = f"[三通道·v15 无多数退v12规则·票型{tally}] " + fb.rationale
                return fb, True
        if key_p2 is not None:
            fb = self._prior_fallback(ans, p2, src="纯LLM票2·v15")
            if fb is not None:
                return fb, True
        ans.rationale += f" ｜[三通道·v15] 法官缺席且无多数（票型 {tally}），维持证据通道原判。"
        return ans, False


    async def _fuse_ensemble(self, ans: Answer, question: str,
                             kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                             candidates: list[str], task_type: str,
                             context: dict | None, n: int) -> tuple[Answer, bool]:
        """v14：把【完整融合判定】独立跑 n 次（每次现抽一路带温度扰动的记忆先验，
        各自走 agree/法官/弃答 分支），再在【最终答案层】众数表决 —— 聚合点=决策点。
        共识不足 SLC_ENS_MIN_CONSENSUS 则诚实弃答（不臆造），保留 v12 缺少的弃权保护。"""
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
        """对 n 次融合输出在答案层计票：弃权单独计；非弃权按 winner/answer 取众数。
        最高共识比例 < 阈值 -> 诚实弃答；否则采纳众数代表判定（保溯源）。"""
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
            new.rationale = (f"[答案层集成·v14 低共识弃答] {n} 次独立融合判定投票 {tally}，"
                             f"最高共识仅 {top_key}={frac:.2f} < 阈值 {min_frac}，诚实弃答（不臆造）。"
                             f"代表判定溯源: {rep_ans.rationale[:300]}")
            return new, True
        new.rationale = (f"[答案层集成·v14] {n} 次独立融合判定投票 {tally}，"
                         f"采纳众数 {top_key}（共识 {frac:.2f}）；代表判定溯源: {rep_ans.rationale[:400]}")
        return new, rep_skip

    async def _fuse(self, ans: Answer, prior: dict | None, question: str,
                    kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                    candidates: list[str], task_type: str,
                    context: dict | None) -> tuple[Answer, bool]:
        """返回 (答案, 是否跳过保形过滤)。跳过保形的仅限 prior/辩论/法官产生的答案——
        保形阈值是按证据通道 conf 语义校准的，套用到先验/票决 conf 上没有统计保证。
        门控：SLC_PRIOR_AGENT 关时 supplied['prior']=None，本函数=直通(v10 行为不变)。
        v12（SLC_LLM_CHIEF=1，随 prior 门控）：诊断显示证据通道独立作答仅 33% 对、
        prior fallback 掷硬币 —— 反转为"LLM 主审"：弃答→证据增强LLM、冲突→LLM法官，
        事实数据(图谱/统计)作为审料注入而非判分。门全关= v11 行为。"""
        if prior is None:
            return ans, False
        agree = ((bucket == "pairwise" and ans.winner and prior.get("winner")
                  and str(ans.winner).upper() == str(prior["winner"]).upper())) or \
                ((bucket == "membership" and ans.answer is not None
                  and prior.get("answer") is not None and bool(ans.answer) == bool(prior["answer"])))
        if not ans.abstained and agree:
            # 两条独立通道同向 = 最强信号（v11 数据：agree 态 77% 对，全系统最高）
            ens = (f"（集成先验一致率={prior.get('agreement')}, "
                   f"{prior.get('n_valid')}/{prior.get('n_draws')} 有效票）"
                   if "agreement" in prior else "")
            ans.rationale += f" ｜[prior·v11 一致] LLM 记忆先验与证据通道同向，零额外成本。{ens}"
            return ans, False

        from knowledge_agents import prior_agent as _pa
        if self._llm_chief():
            facts = self._facts_digest(kg, ck, ans)
            if ans.abstained:
                # 证据弃答 -> 带事实的 LLM 主审重答（而非掷硬币裸 prior）
                aug = await _pa.ask_prior_augmented(question, bucket, candidates, facts)
                if aug is not None and not aug.get("abstained"):
                    fb = self._prior_fallback(ans, aug, src="证据增强LLM·v12")
                    if fb is not None:
                        return fb, True
                fb = self._prior_fallback(ans, prior)   # 增强版弃答/失败 -> 退回裸 prior
                if fb is not None:
                    return fb, True
                return ans, False
            # 已答但冲突 -> 单 LLM 法官看双方论点 + 事实审料
            j = await _pa.adjudicate(question, bucket, candidates,
                                     facts, self._prior_view(prior))
            if j is not None:
                return self._judge_answer(ans, j), True
            # 法官缺席/弃裁 -> prior 优先（证据通道独立准确率≈33%，低于先验）
            fb = self._prior_fallback(ans, prior)
            if fb is not None:
                return fb, True
            return ans, False

        from knowledge_agents import debate as _db

        if ans.abstained:
            # 证据通道弃答 -> 先辩论（先验只是被审观点），无结论再 prior fallback
            if _db.debate_enabled():
                d = await self._debate_wrap(ans, prior, question, kg, ck, bucket,
                                            candidates, task_type, context)
                if d is not None:
                    return d, True
            fb = self._prior_fallback(ans, prior)
            if fb is not None:
                return fb, True
            return ans, False

        # 已作答且与 prior 冲突 -> 辩论仲裁（开启且裁得出才改变；否则证据通道优先）
        if _db.debate_enabled():
            d = await self._debate_wrap(ans, prior, question, kg, ck, bucket,
                                         candidates, task_type, context)
            if d is not None:
                return d, True
        ans.rationale += (" ｜[prior·v11 冲突未仲裁] 记忆先验 "
                          f"{prior.get('winner') or prior.get('answer')}(conf={prior.get('confidence')})"
                          " 与证据通道相左，辩论无定论，证据通道优先（先验不推翻证据）。")
        return ans, False

    # ------------------------------------------------------------------ v12 组件
    @staticmethod
    def _llm_chief() -> bool:
        return os.environ.get("SLC_LLM_CHIEF", "0").lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _facts_digest(kg: BackgroundKnowledge, ck: ComputedKnowledge,
                      ans: Answer) -> str:
        """喂给增强LLM/法官的【事实审料】：只含观测数值与图谱关系，不含管线结论
        （结论会锚定 LLM 判断；lit-rag 堆原文致降的教训 => 结构化摘要 + 控制长度）。"""
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
        except Exception:  # noqa BLE001 —— 审料构造故障 => 空事实（增强LLM退回裸题面）
            facts = {}
        s = _j.dumps(facts, ensure_ascii=False, default=str)
        return s[:1800]

    @staticmethod
    def _prior_view(prior: dict) -> str:
        base = (f"winner={prior.get('winner')} answer={prior.get('answer')} "
                f"conf={prior.get('confidence')} 理由: {str(prior.get('rationale', ''))[:300]}")
        if "agreement" in prior:   # v13 集成先验：把共识度/票型作为可靠性信号交给法官
            base += (f" ｜集成先验: {prior.get('n_valid')}/{prior.get('n_draws')} 有效票, "
                     f"一致率={prior.get('agreement')}, 票型={prior.get('votes')}")
        return base

    @staticmethod
    def _judge_answer(ans: Answer, j: dict) -> Answer:
        """法官结论生成新答案：法官=LLM，conf 语义非证据通道 -> 跳过保形。"""
        import copy
        new = copy.deepcopy(ans)
        new.abstained = False
        new.status = ToolStatus.OK
        new.winner = j.get("winner") if ans.bucket == "pairwise" else None
        new.answer = j.get("answer") if ans.bucket == "membership" else None
        new.confidence = round(float(j.get("confidence", 0.0) or 0.0), 3)
        new.basis = sorted(set(ans.basis) | {"llm_judge"})
        new.decision_mode = f"{ans.decision_mode}+judge"
        new.rationale = (f"[LLM法官·v12 冲突裁决] 证据通道与记忆先验冲突，法官独立复核后 "
                         f"winner={new.winner} answer={new.answer} conf={new.confidence}；"
                         f"法官理由: {str(j.get('rationale', ''))[:240]}。"
                         f"原证据通道判定: winner={ans.winner} answer={ans.answer} "
                         f"conf={ans.confidence}（保留于溯源）。")
        return new

    @staticmethod
    def _prior_fallback(ans: Answer, prior: dict, src: str = "LLM记忆先验") -> Answer | None:
        """证据通道弃答时采纳 LLM 先验答案 —— 来源如实标注，绝不冒充图谱/统计证据。
        src=「LLM记忆先验」（裸 prior）或「证据增强LLM·v12」（事实审料注入后的重答）。"""
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
        ens = (f"（集成先验一致率={prior.get('agreement')}, "
               f"{prior.get('n_valid')}/{prior.get('n_draws')} 有效票, 票型={prior.get('votes')}）"
               if "agreement" in prior else "")
        fb.rationale = (f"[prior智能体·v11 fallback] 证据通道弃答（原文如下），采纳 {src} "
                        f"winner={fb.winner} answer={fb.answer} conf={fb.confidence}{ens}；"
                        f"先验理由: {str(prior.get('rationale', ''))[:200]}。"
                        f"注意：此答案非图谱/统计证据，是记忆先验。 || 原弃答理由: {ans.rationale[:300]}")
        return fb

    async def _debate_wrap(self, ans: Answer, prior: dict, question: str,
                           kg: BackgroundKnowledge, ck: ComputedKnowledge, bucket: str,
                           candidates: list[str], task_type: str,
                           context: dict | None) -> Answer | None:
        """开一轮 3 专家辩论；裁不出结论返回 None。"""
        from knowledge_agents import debate as _db
        digest = (f"证据通道判定: winner={ans.winner} answer={ans.answer} "
                  f"abstained={ans.abstained} conf={ans.confidence} "
                  f"gene_scores={ans.gene_scores}\n判定理由: {ans.rationale[:600]}")
        views = (f"prior(纯LLM记忆先验): winner={prior.get('winner')} answer={prior.get('answer')} "
                 f"conf={prior.get('confidence')} 理由: {str(prior.get('rationale', ''))[:300]}")
        d = await _db.run_debate(question, bucket, candidates, digest, views)
        if d is None:
            return None
        # 辩论结论若与原判定一致 -> 只加确认注记；否则生成新答案（票决 conf 语义不走保形）
        same = ((bucket == "pairwise" and ans.winner and str(d.get("winner", '')).upper()
                 == str(ans.winner).upper()) or
                (bucket == "membership" and ans.answer is not None
                 and d.get("answer") is not None and bool(ans.answer) == bool(d["answer"])))
        if same and not ans.abstained:
            ans.rationale += f" ｜[专家辩论·v11 确认] {d['rationale']} {d['support']}。"
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
        new.rationale = (f"{d['rationale']} {d['support']} 计票={d['tally']}；"
                         f"证据通道原判: winner={ans.winner} answer={ans.answer} "
                         f"abstained={ans.abstained}(conf={ans.confidence})；"
                         f"prior 观点: winner={prior.get('winner')} answer={prior.get('answer')}"
                         f"(conf={prior.get('confidence')})。票决 conf 不走保形（阈值语义不适用）。")
        return new

    # ------------------------------------------------------------------ 保形弃答（calibrated abstention）
    @classmethod
    def _conformal_thresholds(cls) -> dict:
        """从 SLC_CONFORMAL_FILE 指向的 json 读入 {bucket: threshold}（由 conformal 校准产出）。
        未启用/文件缺失 -> 空 dict（不施加，保持原判定）。用模块级全局缓存（不能挂在 pydantic
        Action 类属性上，否则被当 ModelPrivateAttr）。"""
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
            except Exception:  # noqa BLE001 —— 阈值文件损坏即视为未启用，不影响主流程
                data = {}
        _CONFORMAL_CACHE = data
        return data

    # ------------------------------------------------------------------ v18 ③ 选择性 abstention（风险可控旋钮）
    @staticmethod
    def _abstain_theta() -> float:
        """门控 SLC_ABSTAIN_THETA（默认 0 = 关，定版 v20 逐字节）。"""
        try:
            return max(0.0, min(1.0, float(os.environ.get("SLC_ABSTAIN_THETA", "0") or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def _maybe_selective_abstain(self, ans: Answer) -> Answer:
        """终答置信 < θ -> 诚实改弃答（含表决/judge 路径，与保形弃答同语义不同来源：
        保形=校准集控已答错误率；本旋钮=用户选定的风险工作点）。conf 分布已随 tracestore
        落盘，任意 θ 可在 _risk_cover.py 离线 RC 模拟，选点无需重跑臂。"""
        th = self._abstain_theta()
        if th <= 0.0 or ans.abstained:
            return ans
        if float(ans.confidence or 0.0) < th:
            ans.abstained = True
            ans.winner = None
            ans.answer = None
            ans.status = ToolStatus.UNAVAILABLE
            ans.rationale = f"[选择性弃答 conf={float(ans.confidence or 0.0):.2f}<θ{th:.2f}] " + ans.rationale
            ans.basis = sorted(set(ans.basis or []) | {"selective_abstain"})
        return ans

    @classmethod
    def _conformal_filter(cls, ans: Answer, bucket: str) -> Answer:
        """保形保证：置信度低于校准阈值 -> 由"作答"改判为"弃答"（不伪造、不作弊式狂答）。
        阈值来自与测试集不相交的校准集，控制『已答错误率≤α』，故可把 auto 的过度自信收敛到可控风险。"""
        thr = cls._conformal_thresholds().get((bucket or "").lower())
        if thr is None or ans.abstained:
            return ans
        ans.conformal_threshold = thr
        if ans.confidence < thr:
            ans.abstained = True
            ans.winner = None
            ans.answer = None
            ans.status = ToolStatus.UNAVAILABLE
            ans.rationale = (f"[保形弃答 conf={ans.confidence:.2f}<阈值{thr:.2f}] " + ans.rationale)
        return ans
