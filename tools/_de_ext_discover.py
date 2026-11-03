# -*- coding: utf-8 -*-
"""B 路线旁路模块 第1步：为 SKCM / DLBC 找"两组都有重复"的肿瘤内(或血源正常)分组。

只做元数据发现 + 一个样本文件的下载耗时探测，不落整批文件。
发现结果决定后面 DESeq2 能不能跑：
  SKCM  需要 Metastatic 侧文件（本地 manifest 里 6 个全是 Primary Tumor）
  DLBC  需要 Blood Derived Normal 的表达定量文件（GDC 侧此前只看到突变类文件）
"""
import time
import requests
from collections import Counter

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
S = "https://api.gdc.cancer.gov"
H = {"User-Agent": "slc-side-channel/1.0"}

WANT_GROUPS = {
    "TCGA-SKCM": ("Metastatic", "Primary Tumor"),
    "TCGA-DLBC": ("Primary Tumor", "Blood Derived Normal"),
    "TCGA-OV": ("Primary Tumor", "Recurrent Tumor"),
}


def sample_of(x):
    for c in (x.get("cases") or []):
        s = c.get("samples") or []
        if s:
            return s[0]
    return {}


def files_for(proj, sample_types):
    body = {"filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [proj]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": list(sample_types)}},
    ]}, "fields": "file_id,file_name,file_size,cases.samples.sample_type",
        "size": 200, "format": "json"}
    r = requests.post(S + "/files", json=body, proxies=P, headers=H, timeout=120)
    if r.status_code != 200:
        print(f"   {proj} HTTP {r.status_code} {r.text[:150]}")
        return []
    d = r.json()
    rows = d["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    return rows


for proj, (g1, g2) in WANT_GROUPS.items():
    rows = files_for(proj, (g1, g2))
    c = Counter(sample_of(x).get("sample_type") for x in rows)
    print(f"== {proj}  候选组 {g1!r} / {g2!r}  ->  {dict(c)}   (返回 {len(rows)} 条)")
    sizes = [x.get("file_size") for x in rows if x.get("file_size")]
    if sizes:
        print(f"      file_size 中位={int(sorted(sizes)[len(sizes)//2])/1e6:.2f}MB  "
              f"最小={int(min(sizes))/1e6:.2f}MB")
    for st in (g1, g2):
        sub = [x for x in rows if sample_of(x).get("sample_type") == st][:2]
        for x in sub:
            print(f"      {st:<22} {x['file_id']}  {x['file_name'][:60]}")

# 下载速度探测：拿一个 SKCM Metastatic 文件的前 2MB
print("\n=== 下载速度探测（GET /data/<id>，最多读 3MB）===")
rows = files_for("TCGA-SKCM", ("Metastatic",))
if rows:
    fid = rows[0]["file_id"]
    t0 = time.time()
    n = 0
    with requests.get(f"{S}/data/{fid}", stream=True, proxies=P, headers=H,
                      timeout=(15, 120)) as r:
        print(f"   HTTP {r.status_code}  Content-Length={r.headers.get('Content-Length')}")
        for chunk in r.iter_content(chunk_size=1 << 20):
            n += len(chunk)
            if n >= 3 * (1 << 20):
                break
        r.close()
    dt = time.time() - t0
    print(f"   到手 {n/1e6:.2f}MB / {dt:.1f}s = {n/1e6/max(dt,.001):.2f} MB/s"
          f"  ->  一个 4.2MB 文件约需 {4.2/max(n/1e6/dt,.001):.0f}s")
else:
    print("   没有 Metastatic GEQ 文件，SKCM 肿瘤内对比不可构造")
