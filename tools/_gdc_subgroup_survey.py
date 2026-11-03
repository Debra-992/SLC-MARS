# -*- coding: utf-8 -*-
"""肿瘤内对比的可行性：TCGA-OV / TCGA-SKCM 的 Gene Expression Quantification 文件
按 sample_type 的完整分布（不截断），决定能构造哪种"两组都有重复"的对比。"""
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"


def sample_of(x):
    for c in (x.get("cases") or []):
        s = c.get("samples") or []
        if s:
            return s[0]
    return {}


for proj in ("TCGA-OV", "TCGA-SKCM", "TCGA-DLBC"):
    body = {"filters": {"op": "and", "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [proj]}},
                {"op": "in", "content": {"field": "data_type",
                                         "value": ["Gene Expression Quantification"]}}]},
            "fields": "file_id,file_name,cases.samples.sample_type",
            "size": 2000, "format": "json"}
    r = requests.post(S + "/files", json=body, proxies=P, timeout=120)
    rows = r.json()["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    c = Counter(sample_of(x).get("sample_type") for x in rows)
    print(f"== {proj}  GEQ 文件 {sum(c.values())} 个（本页 {len(rows)}）")
    for k, v in c.most_common():
        print(f"      {v:5d}  {k}")
    print()
