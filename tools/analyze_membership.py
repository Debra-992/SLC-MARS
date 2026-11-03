# -*- coding: utf-8 -*-
"""membership 桶诊断：把 v8 heuristic 的 membership 判定按"驱动来源"拆开，定位错误来源。

membership 判定内核（decision_action._heuristic）：net = pol + W_BUFFER*buf
  - pol∈{-1,0,+1} 因果方向（RelaEvent ⊗ 题面动词）
  - buf=-redundancy∈[-1,0] 旁系缓冲（单边，只削弱阳性）
  => answer=True 仅可能来自 pol=+1；answer=False 来自 pol=-1 或"pol=0 仅缓冲"；net==0 弃答。
本脚本从 calibstore/tracestore 的 rationale 里正则抽回 (pol,buf,net)，
交叉 预测答案 vs 真值，按驱动分类统计正确率，并给出癌种/是否有DE面板/真值基率的分布。
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict

NET = re.compile(r"net=([+-]?\d*\.?\d+)")
POL = re.compile(r"pol=([+-]\d+)")
BUF = re.compile(r"buf=([+-]?\d*\.?\d+)")


def rows_from(path):
    out = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def heur_membership(entry):
    """返回 membership 题的 heuristic 已答样本 dict，或 None。"""
    if entry.get("bucket") != "membership" or entry.get("error"):
        return None
    sysobj = (entry.get("systems") or {}).get("pipeline_heuristic") or {}
    ans = sysobj.get("answer") or {}
    outcome = sysobj.get("outcome")
    if outcome not in ("correct", "wrong"):
        return None  # 弃答/异常不计
    rat = ans.get("rationale", "") or ""
    g = (entry.get("effective_genes") or [""])[0]
    gt = bool((entry.get("ground_truth") or {}).get("answer"))
    pred = bool(ans.get("answer"))
    m_pol, m_buf = POL.search(rat), BUF.search(rat)
    pol = int(m_pol.group(1)) if m_pol else 0
    buf = float(m_buf.group(1)) if m_buf else 0.0
    red = float((entry.get("evidence_view") or {}).get("paralog_redundancy", {}).get(g, 0.0) or 0.0)
    has_de = bool((entry.get("evidence_view") or {}).get("de"))
    cancer = None
    q = entry.get("qid", "")
    parts = q.split("_")
    if len(parts) >= 3:
        cancer = parts[2]
    return dict(pred=pred, gt=gt, outcome=outcome, pol=pol, buf=buf, red=red,
                has_de=has_de, cancer=cancer, conf=float(ans.get("confidence") or 0.0))


def driver(r):
    if r["pol"] == 1:
        return "pol=+1 -> 判True(因果促进)"
    if r["pol"] == -1:
        return "pol=-1 -> 判False(因果抑制)"
    if r["pol"] == 0:
        # 已答且 pol=0 => net=0.8*buf<0，即仅凭(哪怕≈0的)旁系缓冲被判 nonessential
        return "pol=0 仅缓冲 -> 判False"
    return "其它"


def rate(n, c):
    return f"{c}/{n}={c/n:.1%}" if n else "0/0"


def main():
    calib = sys.argv[1] if len(sys.argv) > 1 else r"outputs\traces\calibstore.jsonl"
    data = [d for d in (heur_membership(e) for e in rows_from(calib)) if d]
    print(f"membership 已答样本 n={len(data)}")
    if not data:
        return

    gt_pos = sum(1 for r in data if r["gt"])
    gt_neg = len(data) - gt_pos
    majority = "True" if gt_pos >= gt_neg else "False"
    print("\n[真值基率] essential(True)=%d / nonessential(False)=%d" % (gt_pos, gt_neg))
    print("[多数类常数预测器参照准确率] 全判 %s -> %.1f%%"
          % (majority, 100.0 * max(gt_pos, gt_neg) / len(data)))

    print("\n[驱动来源 × 正确率]")
    bydrv = defaultdict(lambda: [0, 0])
    for r in data:
        d = driver(r)
        bydrv[d][0] += 1
        bydrv[d][1] += (r["outcome"] == "correct")
    for d, (n, c) in sorted(bydrv.items(), key=lambda kv: -kv[1][0]):
        print(f"  {d:<26} {rate(n, c)}")

    print("\n[预测 vs 真值 混淆矩阵] (行=pred, 列=gt)")
    cm = Counter((r["pred"], r["gt"]) for r in data)
    for pred in (True, False):
        for gt in (True, False):
            tag = "对" if pred == gt else "错"
            print(f"  pred={pred!s:<5} gt={gt!s:<5} -> {cm.get((pred,gt),0):>3}  {tag}")

    print("\n[按是否有 DE 面板(旁系缓冲能否生效)]")
    for hd in (True, False):
        sub = [r for r in data if r["has_de"] == hd]
        c = sum(1 for r in sub if r["outcome"] == "correct")
        print(f"  has_de={hd!s:<5} {rate(len(sub), c)}")

    print("\n[按癌种 正确率]")
    byc = defaultdict(lambda: [0, 0])
    for r in data:
        byc[r["cancer"]][0] += 1
        byc[r["cancer"]][1] += (r["outcome"] == "correct")
    for cancer, (n, c) in sorted(byc.items(), key=lambda kv: (-kv[1][0], kv[0])):
        print(f"  {str(cancer):<12} {rate(n, c)}")

    print("\n[缓冲强度 redundancy × 预测False 是否过头]")
    bufs = [r for r in data if r["pol"] == 0 and r["buf"] < 0]
    if bufs:
        hi = [r for r in bufs if r["red"] >= 0.99]
        print(f"  纯缓冲判False 共 {len(bufs)}，其中 redundancy=1.0 饱和 {len(hi)}，"
              f"这些判False且真值也是False 的 {sum(1 for r in bufs if not r['gt'])}/{len(bufs)}")
        wrongneg = sum(1 for r in bufs if r["gt"])
        print(f"  -> 真值其实 essential=True 却被缓冲判 False（假阴）= {wrongneg}/{len(bufs)}")


if __name__ == "__main__":
    main()
