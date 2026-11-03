# -*- coding: utf-8 -*-
"""TARGET 白血病项目里，DESeq2 需要的 RNA-seq 计数文件到底存不存在？
（白血病系题目占 topK 11 题里的多数，本项目 TCGA_CODES 不含 TARGET-* → 一直被判 UnsupportedCancer。
 若 TARGET 里根本没有 DESeq2 能吃的计数文件，那"扩展 cancer_config 支持 TARGET"也是死路。）"""
import requests

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "api.gdc.cancer.gov"
PROJECTS = ["TARGET-ALL-P1", "TARGET-ALL-P2", "TARGET-ALL-P3", "TARGET-NBL-1", "TARGET-RTMB-1"]


def analyze(path, project):
    body = {"filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}}]}}
    r = requests.post(f"https://{S}/{path}", json=body, proxies=P, timeout=40)
    if r.status_code != 200:
        return f"HTTP {r.status_code} {r.text[:80]}"
    return r.json()


def files_count(project, data_type):
    body = {"filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}}]},
        "size": 3, "format": "json"}
    r = requests.post(f"https://{S}/files", json=body, proxies=P, timeout=40)
    d = r.json()
    rows = d["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    total = (d["data"].get("pagination") or {}).get("total")
    return total, [x.get("file_name") for x in rows[:2]]


for pr in PROJECTS:
    print("=" * 88)
    print(pr)
    dt = analyze("analyze/data/data_types", pr)
    if isinstance(dt, dict):
        for x in sorted(dt, key=lambda y: -y["count"])[:12]:
            print(f"      {x['count']:6d}  {x['name']}")
    else:
        print("      ", dt)
    for want in ("RNA-Seq", "Transcriptome Folding Quantification",
                 "Aligned Reads", "Aligned Coordinate Sorted Reads (BAM Level 3)"):
        total, names = files_count(pr, want)
        print(f"   files[{want!r}] total={total}  样例={names}")
