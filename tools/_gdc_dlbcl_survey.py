# -*- coding: utf-8 -*-
"""决定性复查：NCICCR-DLBCL / 其他 DLBCL 研究里有没有【正常淋巴组织】的原始计数？
用正确的 GDC 过滤字段（cases.samples.sample_type 在 file 侧）分别数：
  - 每个项目的 sample_type 分布
  - 每个项目的 data_type 分布（看有没有 Gene Expression Quantification / star_counts）
"""
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"


def post(path, body):
    r = requests.post(S + path, json=body, proxies=P, timeout=120)
    d = r.json()
    rows = d["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    return r.status_code, rows, (d["data"].get("pagination") or {})


def sample_of(x):
    for c in (x.get("cases") or []):
        s = c.get("samples") or []
        if s:
            return s[0]
    return {}


for proj in ("NCICCR-DLBCL", "TCGA-DLBC", "CGAC-DLBC", "TCGA-LAML", "CTRP-DLBC"):
    body = {"filters": {"op": "in", "content": {
                "field": "cases.project.project_id", "value": [proj]}},
            "fields": "file_id,file_name,data_type,cases.samples.sample_type,"
                      "cases.samples.tissue_site,cases.disease_type",
            "size": 2000, "format": "json"}
    st, rows, pg = post("/files", body)
    if st != 200:
        print(f"{proj}: HTTP {st}")
        continue
    stc = Counter((sample_of(x).get("sample_type"), x.get("data_type")) for x in rows)
    print(f"== {proj}  文件总数={pg.get('total')} 本页={len(rows)}")
    for k, v in sorted(stc.items(), key=lambda y: -y[1])[:10]:
        print(f"      {v:5d}  sample_type={str(k[0]):<45} data_type={k[1]}")
    normals = [x for x in rows
               if sample_of(x).get("sample_type") == "Solid Tissue Normal"
               and x.get("data_type") == "Gene Expression Quantification"]
    print(f"      >>> normal × Gene Expression Quantification = {len(normals)}")
