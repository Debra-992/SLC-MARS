# -*- coding: utf-8 -*-
"""B 路线第二问：CPTAC-3 等泛癌项目的 normal 文件里，有没有【卵巢 / 皮肤 / 淋巴-血液】部位？
决定 OV / SKCM / DLBC 三道题能否用 GDC 内的组织匹配正常样本喂 DESeq2。
顺带把 TCGA-DLBC 的全部文件（不限 sample_type）列出来，确认它到底有没有正常对照。
"""
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"


def post(path, body):
    r = requests.post(S + path, json=body, proxies=P, timeout=120)
    print(f"   {path} -> HTTP {r.status_code}")
    d = r.json()
    rows = d["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    return rows


def sample_of(x):
    for c in (x.get("cases") or []):
        s = c.get("samples") or []
        if s:
            return s[0]
    return {}


WANT = ("CPTAC-3", "TCGA-OV", "TCGA-SKCM", "TCGA-DLBC", "TCGA-LAML", "TCGA-CLLE",
        "TARGET-NBL-1", "EEA-CCDT-2", "CTRP-LUCAP")
body = {"filters": {"op": "and", "content": [
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "cases.project.project_id", "value": list(WANT)}}]},
    "fields": "file_id,file_name,cases.project.project_id,cases.samples.sample_type,"
              "cases.samples.tissue_site,cases.samples.description,cases.disease_type",
    "size": 2000, "format": "json"}
rows = post("/files", body)
print(f"共 {len(rows)} 条文件\n")

per = {}
for x in rows:
    s = sample_of(x)
    proj = ((x.get("cases") or [{}])[0].get("project") or {}).get("project_id")
    key = (proj, s.get("sample_type"), s.get("tissue_site"), s.get("disease_type"))
    per.setdefault(key, []).append(x["file_id"])

print(f"{'项目':<14} {'sample_type':<22} {'部位':<20} {'疾病':<26} 文件数")
print("-" * 100)
for k, v in sorted(per.items(), key=lambda y: (str(y[0][0]), -len(y[1]))):
    print(f"{str(k[0]):<14} {str(k[1]):<22} {str(k[2]):<20} {str(k[3])[:24]:<26} {len(v):>4d}")

print("\n>>> 组织匹配候选（normal 侧）：")
for tissue_kw in ("ovary", "fallopian", "skin", "lymph", "blood", "bone marrow", "spleen"):
    hit = [(k, v) for k, v in per.items()
           if k[1] == "Solid Tissue Normal" and tissue_kw in str(k[2]).lower() + str(k[3]).lower()]
    n = sum(len(v) for _k, v in hit)
    srcs = ", ".join(sorted({str(k[0]) for k, _ in hit})) or "-"
    print("   %-12s 文件数=%-4d 来源=%s" % (tissue_kw, n, srcs))
