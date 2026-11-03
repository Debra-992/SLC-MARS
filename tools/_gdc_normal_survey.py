# -*- coding: utf-8 -*-
"""B 路线可行性第一问：GDC 里哪些项目有【正常组织】的 STAR 原始计数（unstranded）？
只要有 ≥2 个 normal 的 Gene Expression Quantification 文件的项目，才能喂 DESeq2。
经本地 SOCKS 代理，纯元数据查询，不下载文件。
"""
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"

body = {
    "filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "data_type",
                                 "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type",
                                 "value": ["Solid Tissue Normal", "Recurrent Blood Normal",
                                           "Additional - New Primary"]}},
    ]},
    "fields": "file_id,file_name,cases.project.project_id,cases.samples.sample_type,"
              "cases.samples.tissue_site,cases.disease_type",
    "size": 2000, "format": "json",
}
r = requests.post(S + "/files", json=body, proxies=P, timeout=120)
print("HTTP", r.status_code)
d = r.json()
rows = d["data"]["hits"]
if isinstance(rows, dict):
    rows = rows.get("hits", [])
print("returned", len(rows), "pagination =", d["data"].get("pagination"))


def first(case, key):
    """cases 是 list；samples 在 case 里也可能是 dict 或 list。"""
    cs = case if isinstance(case, list) else [case]
    for c0 in cs:
        s = (c0 or {}).get(key)
        if s is None:
            continue
        return s[0] if isinstance(s, list) and s else s
    return {}


c = Counter()
for x in rows:
    case = x.get("cases") or {}
    proj = (((first(case, "project")) or {}).get("project_id"))
    st = ((first(case, "samples") or {}).get("sample_type"))
    ts = ((first(case, "samples") or {}).get("tissue_site"))
    c[(proj, st, ts)] += 1
print("按 (项目, sample_type, tissue_site) 计数：")
for k, v in sorted(c.items(), key=lambda y: -y[1])[:45]:
    print(f"   {v:4d}  {k[0]:<14} {str(k[1]):<22} {k[2]}")
