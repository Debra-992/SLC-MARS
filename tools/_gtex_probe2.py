# -*- coding: utf-8 -*-
"""定位 GTEx 可用入口：API 路径变体 + 静态分析文件（只探大小，不落盘）。"""
import requests

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}
H = {"User-Agent": "slc-netcheck/1.0", "Accept-Encoding": "identity"}

CAND = [
    ("GET", "https://gtexportal.org/API/tissue/"),
    ("GET", "https://gtexportal.org/API/tissue/Spleen"),
    ("GET", "https://gtexportal.org/API/expression/median/SLC39A9/Ovary"),
    ("GET", "https://gtexportal.org/API/expression/median?geneId=SLC39A9&tissueSiteDetailId=Ovary"),
    ("GET", "https://storage.googleapis.com/gtex_analysis_v8/single_cell_sec._data/mock_file"),
    ("GET", "https://storage.googleapis.com/gtex_analysis_v8/"
            "rnaseq_fasta/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"),
]
for method, url in CAND:
    try:
        r = requests.request(method, url, proxies=P, headers=H, timeout=45, stream=True,
                             allow_redirects=False)
        cl = r.headers.get("Content-Length") or r.headers.get("X-Goog-TotalLength") or "?"
        ct = r.headers.get("Content-Type")
        loc = r.headers.get("Location")
        head = r.raw.read(200, decode_content=False) if r.status_code < 400 else b""
        print(f"HTTP {r.status_code} len={cl} type={ct} loc={loc}\n     url={url}\n     head={head[:160]!r}")
    except Exception as e:
        print(f"{type(e).__name__}: {e}\n     url={url}")
    print("-" * 88)
