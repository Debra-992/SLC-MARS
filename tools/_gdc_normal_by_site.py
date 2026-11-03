# -*- coding: utf-8 -*-
"""B 路线落点确认：GDC 的 Solid Tissue Normal GEQ 文件（1933 个）按【疾病类型】怎么分布？
GDC 的 file 记录里 cases.samples 拿不到 disease_type，改用 /cases 端点按正常样本反查。
只查元数据，不下载。
"""
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"

# 从 cases 侧看：带 Solid Tissue Normal 样本的项目 → 疾病类型
body = {"filters": {"op": "and", "content": [
    {"op": "==", "content": {"field": "samples.sample_type", "value": "Solid Tissue Normal"}}]},
    "fields": "disease_type,primary_site,project.project_id,sample_type",
    "size": 2000, "format": "json"}
r = requests.post(S + "/cases", json=body, proxies=P, timeout=120)
print("POST /cases -> HTTP", r.status_code)
d = r.json()
rows = d["data"]["hits"]
if isinstance(rows, dict):
    rows = rows.get("hits", [])
print("returned", len(rows), "pagination =", d["data"].get("pagination"), "\n")

site = Counter()
proj_site = Counter()
for x in rows:
    ps = x.get("primary_site")
    dt = x.get("disease_type")
    pj = (x.get("project") or {}).get("project_id")
    if isinstance(ps, list):
        ps = ",".join(ps)
    if isinstance(dt, list):
        dt = ",".join(dt)
    site[ps] += 1
    proj_site[(pj, ps)] += 1

print("=== 带正常样本的病例，按 primary_site 计数（前 30）===")
for k, v in site.most_common(30):
    print(f"   {v:5d}  {k}")

print("\n=== 我们关心的三个部位 ===")
for kw in ("ovary", "fallopian", "skin", "lymph node", "blood", "bone marrow", "spleen", "uterus"):
    hits = {k: v for k, v in proj_site.items() if kw in str(k[1]).lower()}
    tot = sum(hits.values())
    srcs = ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(hits.items(), key=lambda y: -y[1])[:6])
    print(f"   {kw:<13} 病例数={tot:<5} {srcs}")
