# -*- coding: utf-8 -*-
"""选项1 网络可达性复测（经本地 SOCKS 代理）—— 用项目真实的调用形状验证。

上一轮判"GDC 不可达"的两个证据都不成立：
  (1) 用的是 `GET /data_files`，该资源现版根本没有（POST 也 404）→ 端点用错，不是网络问题；
  (2) 直连确实被出口拦（连 google 都 `Remote end closed connection`），但 Windows
      注册表里配着活着的 SOCKS 代理 127.0.0.1:1080，环境变量里没有 HTTP(S)_PROXY，
      所以 requests/urllib 全走了被拦的直连路。

本脚本走 mcp-venv（项目 tcga_download_tool 所在的解释器，已装 PySocks+requests），
用 5 个判别性探针回答"选项1 到底可行吗"：
  1 GET  /status          —— 控制组，确认代理通道对 GDC 有效
  2 POST /files           —— 正确的文件枚举端点，看 TCGA-OV 有没有 Solid Tissue Normal
  3 POST /files 计数       —— 同样查 SKCM / DLBC
  4 GET  /data/<uuid>     —— TCGA.py:178 的真下载 URL 形状（流式读 200 字节）
  5 set_curl_proxy + POST —— TCGAbiolinks 依赖的 R `download.file(method="libcurl")`
                             是否吃 *_PROXY 环境变量（决定 R 路能不能一起打通）

不写盘、不整文件下载。用法：
  outputs\\mcp-venv\\Scripts\\python.exe tools\\_gdc_net_recheck2.py
"""
from __future__ import annotations
import json, os, subprocess, sys

import requests

SOCKS = "socks5h://127.0.0.1:1080"          # socks5h = 域名也交给代理解析（本地 DNS 被污染时用）
PROXIES = {"http": SOCKS, "https": SOCKS}
API = "https://api.gdc.cancer.gov"
T = 40


def q(project, extra=None):
    f = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_format", "value": ["TSV"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
    ]}
    if extra:
        f["content"].extend(extra)
    return {"filters": f, "fields": "file_id,file_name,sample_type,analysis.workflow_type",
            "size": 400, "format": "json"}


def probe_status():
    r = requests.get(API + "/status", proxies=PROXIES, timeout=T)
    print(f"[1] GET /status -> HTTP {r.status_code}  {r.text[:160]}")
    return r.status_code == 200


def probe_files(project):
    r = requests.post(API + "/files", json=q(project), proxies=PROXIES, timeout=T)
    if r.status_code != 200:
        print(f"[2] POST /files {project} -> HTTP {r.status_code} {r.text[:200]}")
        return None
    d = r.json()
    # POST 返回扁平 data.hits 列表 + data.pagination；GET 才是 {total, hits}
    hits = d["data"]["hits"]
    if isinstance(hits, dict):
        total, rows = hits.get("total"), hits.get("hits", [])
    else:
        total, rows = (d["data"].get("pagination") or {}).get("total"), hits
    from collections import Counter
    c = Counter((h.get("sample_type"), (h.get("analysis") or {}).get("workflow_type")) for h in rows)
    print(f"[2] POST /files {project}: total={total} returned={len(rows)}")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"        {v:4d}  sample_type={k[0]!r:26s} workflow={k[1]!r}")
    norms = [h for h in rows if h.get("sample_type") == "Solid Tissue Normal"]
    stars = [h for h in rows if str(h.get("file_name", "")).endswith(".rna_seq.star_counts.tsv")
             or h.get("workflow_type") == "STAR - RNA-seq"
             or (h.get("analysis") or {}).get("workflow_type") in ("STAR - RNA-seq", "Star - RNA Sequencing")]
    print(f"        → Solid Tissue Normal 到手 {len(norms)} 条 | 疑似 star_counts {len(stars)} 条 "
          f"| 首个 file_name={rows[0].get('file_name') if rows else '-'}")
    return rows


def probe_download(rows):
    fid = rows[0]["file_id"]
    url = f"{API}/data/{fid}"                       # 与 TCGA.py:178 完全同形
    try:
        with requests.get(url, stream=True, timeout=(10, 60), proxies=PROXIES) as resp:
            print(f"[4] GET /data/<uuid> -> HTTP {resp.status_code} "
                  f"Content-Type={resp.headers.get('Content-Type')} "
                  f"Content-Disposition={str(resp.headers.get('Content-Disposition'))[:60]}")
            head = b""
            for chunk in resp.iter_content(chunk_size=1024):
                head += chunk
                if len(head) > 200:
                    break
            resp.close()
            print(f"        真字节到手 {len(head)}B，首行: {head.splitlines()[0][:120]!r}")
            return True
    except Exception as e:
        print(f"[4] GET /data/<uuid> FAILED {type(e).__name__}: {e}")
        return False


def probe_normal_counts(project):
    """file 实体没有 sample_type 属性，必须按 cases.samples.sample_type 过滤才能数正常样本。"""
    f = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type",
                                 "value": ["Solid Tissue Normal"]}},
    ]}
    body = {"filters": f, "fields": "file_id,file_name,cases.samples.sample_type",
            "size": 5, "format": "json"}
    r = requests.post(API + "/files", json=body, proxies=PROXIES, timeout=T)
    d = r.json()
    rows = d["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    total = (d["data"].get("pagination") or {}).get("total")
    print(f"[3] {project} Solid Tissue Normal 的 Gene Expression Quantification 文件数 = {total}")
    for h in rows[:3]:
        print(f"        {h.get('file_id')}  {h.get('file_name')}")
    return rows


def probe_r_proxy():
    print("[5] R libcurl 是否吃 *_PROXY 环境变量（TCGAbiolinks 路）")
    env = dict(os.environ)
    env["ALL_PROXY"] = SOCKS
    env["HTTP_PROXY"] = env["HTTPS_PROXY"] = SOCKS
    script = (
        'options(timeout=25)\n'
        'url <- "https://api.gdc.cancer.gov/status"\n'
        'res <- httr::GET(url)\n'
        'cat("R_HTTR_STATUS=", httr::status_code(res), "\\n")\n')
    script2 = (
        'options(timeout=25)\n'
        'con <- file("https://api.gdc.cancer.gov/status", open="rb")\n'
        'ls <- readLines(con, warn=FALSE)\n'
        'close(con)\n'
        'cat("R_FILE_LINES=", length(ls), "\\n", sep="")\n'
        'cat(substr(paste(ls, collapse=" "), 1, 120), "\\n")\n')
    for tag, sc in (("file()+*_PROXY", script2), ("httr+*_PROXY", script)):
        try:
            p = subprocess.run(["Rscript", "-e", sc], env=env, capture_output=True,
                               text=True, timeout=180)
            out = (p.stdout or "").strip().replace("\n", " | ")
            print(f"        [{tag}] rc={p.returncode} out={out[:260]}")
            if p.returncode != 0:
                print(f"        [{tag}] err={(p.stderr or '')[:260]}")
        except Exception as e:
            print(f"        [{tag}] {type(e).__name__}: {e}")


def main():
    print("=" * 92)
    ok = probe_status()
    if not ok:
        print("代理通道对 GDC 无效，后面不必测")
        return
    rows = None
    for proj in ("TCGA-OV", "TCGA-SKCM", "TCGA-DLBC"):
        n = probe_normal_counts(proj)
        if n and rows is None:
            rows = n
    if rows:
        probe_download(rows)
    probe_r_proxy()
    print("=" * 92)


if __name__ == "__main__":
    main()
