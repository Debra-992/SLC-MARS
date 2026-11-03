"""检查 Neo4j + 尝试一次 PPR 最小突击（单题），验证 compute 链路后再批量导出 11 题候选。
只读，不写 tracestore，不与 compare_eval 冲突。"""
import os, sys, json, asyncio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Step 1: env check
for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]:
    print(f"  {k}={'****' if k.endswith('PASSWORD') else os.environ.get(k,'(not set)')}")

# Step 2: try Neo4j ping
try:
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    with GraphDatabase.driver(uri, auth=(user, pwd)) as drv:
        drv.verify_connectivity()
    print("  Neo4j 连接 OK")
except Exception as e:
    print(f"  Neo4j 连接失败: {e}")
    print("  => 将仅产出静态数据结构（题面+GT），PPR 候选列留空、待 Neo4j 可达时再填。")

# Step 3: 加载 11 道 topk 题 + GT
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

qs = [r for r in load_jsonl(ROOT / "data" / "questions_clean.jsonl") if r.get("task_bucket") == "topk"]
gts = {r["qid"]: r.get("ground_truth", []) for r in load_jsonl(ROOT / "evaluation" / "ground_truth_clean.jsonl")}

print(f"\n  加载 topk 题: {len(qs)} 道, GT 覆盖: {sum(1 for q in qs if q['qid'] in gts)}")

# Step 4: 尝试跑一道 topk 的 compute（仅 PPR 候选）
print("\n--- 尝试跑第一道 topk 的 PPR 候选 ---")
q0 = qs[0]
print(f"  qid={q0['qid']}")
print(f"  cancer={q0.get('constraints',{}).get('cancer')}")
print(f"  GT ranking={gts.get(q0['qid'], [])}")

async def try_one():
    from knowledge_agents.roles.compute_role import ComputeRole
    from knowledge_agents.schema import parse_knowledge, ComputedKnowledge
    role = ComputeRole()
    constraints = q0.get("constraints", {})
    cancer = constraints.get("cancer", "")
    role.set_task(qid=q0["qid"], gene=None, genes=[], cancer=cancer, task_bucket="topk")
    await role.run(f"gene=None cancer={cancer} bucket=topk {q0['question']}")
    ck = parse_knowledge(role.rc.memory.get()[-1].content)
    assert isinstance(ck, ComputedKnowledge)
    gc = ck.graph_computation
    if gc:
        print(f"  PPR status={gc.status.value} seeds={gc.seed_count} nodes={gc.node_count}")
        candidates = [(c.gene, c.degree_corrected_score) for c in (gc.candidates or [])]
        print(f"  PPR 候选数: {len(candidates)}")
        print(f"  Top-20 genes: {[g for g,_ in candidates[:20]]}")
        gt = gts.get(q0["qid"], [])
        gt_ranks = {g: i for i, g in enumerate(gt)}
        top20 = [g for g, _ in candidates[:20]]
        gt_in_top20 = [g for g in gt if g in top20]
        print(f"  GT genes in PPR top-20: {gt_in_top20} ({len(gt_in_top20)}/5)")
        # check ranks of GT in full PPR list
        all_genes = [g for g, _ in candidates]
        for g in gt:
            idx = all_genes.index(g) if g in all_genes else -1
            print(f"    GT {g}: PPR rank {idx+1 if idx>=0 else 'not found'}")
        return candidates
    else:
        print("  PPR 未运行 (无 graph_computation)")
        return None

cands = asyncio.run(try_one())

if cands:
    print("\n  PPR 候选获取成功。可以批量导出全部 11 题。")
else:
    print("\n  PPR 候选获取失败 — 仅能产出静态数据结构。")