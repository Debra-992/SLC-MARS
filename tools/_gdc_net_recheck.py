# -*- coding: utf-8 -*-
"""选项1 网络可达性复测：按【真实下载链路】的 URL 形状逐端点探测。

上一轮我只试了 `GET /data_files`（该端点只接受 POST，404 属预期）就判了不可达，证据不足。
本脚本覆盖项目代码真正调用的每一条形：
  A. Python 侧 TCGA.py:178  ->  https://api.gdc.cancer.gov/data/<file_id>  （stream 下载真身）
  B. GDC POST /status        ->  真 JSON 信封（status/code/data 三键齐全，不是代理假页）
  C. GDC POST /cases         ->  项目实际用来枚举样本的 _default_fields_（POST + JSON body）
  D. GDC POST /data_files    ->  合法形式的 data_files 查询（旧探测用 GET 才 404）
  E. legacy  api.gdc.cancer.gov/legacy/api/projects/TCGA-X
  F. 代理指纹：响应 Server / Via 头，判断是否被 egress 代理中间人

只发元数据级小请求 + 对下载端点做 HEAD/Range 的 1 字节读，不落盘、不拉大文件。
用法： .venv\\Scripts\\python.exe tools\\_gdc_net_recheck.py
"""
from __future__ import annotations
import json, socket, ssl, sys, urllib.error, urllib.request

FILE_ID = "0bb8cf8a-1e2c-4bbb-9c3e-6b6a5b0b0b0b"   # 仅探路，任意 UUID（404 vs 拒绝 才有判别力）
UA = {"User-Agent": "slc-netcheck/1.0", "Accept": "application/json"}
TIMEOUT = 20


def show(tag, url, method="GET", body=None, headers=None):
    h = dict(UA)
    if headers:
        h.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    print(f"\n----- {tag}\n        {method} {url}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read(400)
            hv = dict(r.headers)
            print(f"        HTTP {r.status}  |  Server={hv.get('Server')}  Via={hv.get('Via')}  "
                  f"Content-Type={hv.get('Content-Type')}  Len={hv.get('Content-Length')}")
            print(f"        body[:200]: {raw[:200]!r}")
    except urllib.error.HTTPError as e:
        raw = e.read(300) if e.fp else b""
        hv = dict(e.headers or {})
        print(f"        HTTPError {e.code} {e.reason}  |  Server={hv.get('Server')}  Via={hv.get('Via')}")
        print(f"        body[:200]: {raw[:200]!r}")
    except Exception as e:
        print(f"        {type(e).__name__}: {e}")


def main():
    print("=" * 96)
    print("0) DNS / TCP / TLS 分层")
    host = "api.gdc.cancer.gov"
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        print(f"   DNS {host} -> {sorted({i[4][0] for i in infos})}")
    except Exception as e:
        print(f"   DNS FAILED: {type(e).__name__}: {e}")
        return
    for fam, socktype, proto, canon, sa in infos[:2]:
        try:
            s = socket.create_connection(sa, timeout=10)
            s.close()
            print(f"   TCP {sa} : OK")
        except Exception as e:
            print(f"   TCP {sa} : {type(e).__name__}: {e}")
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ts:
                cert = ts.getpeercert()
                print(f"   TLS  OK  issued_to={dict((k, v) for f, k, v in cert['subject'])}")
    except Exception as e:
        print(f"   TLS  {type(e).__name__}: {e}")

    print("\n" + "=" * 96)
    print("1) GDC 主 API —— 项目代码真实调用的条形")
    show("B. status（上一轮说它 200，这次验证是不是真 JSON 信封）",
         "https://api.gdc.cancer.gov/status", method="POST", body={})
    show("C. cases 查询（TCGAbiolinks/GDC 枚举样本用的 _default_fields_）",
         "https://api.gdc.cancer.gov/cases", method="POST",
         body={"filters": {"op": "and", "content": [
                  {"op": "==", "content": {"field": "project.project_id", "value": "TCGA-OV"}}]},
               "fields": "_default_fields_", "size": 1, "format": "json"})
    show("D. data_files POST（上一轮误用 GET 才 404）",
         "https://api.gdc.cancer.gov/data_files", method="POST",
         body={"filters": {"op": "==", "content": {
                   "field": "cases.project.project_id", "value": "TCGA-OV"}},
               "fields": "file_id,file_name,sample_type", "size": 3, "format": "json"})
    show("A. 真下载端点 = TCGA.py:178 的 URL 形状（HEAD，不拉体）",
         f"https://api.gdc.cancer.gov/data/{FILE_ID}", method="HEAD")
    show("A2. 同一端点 Range 请求（有些代理放行 GET 但拦 HEAD）",
         f"https://api.gdc.cancer.gov/data/{FILE_ID}",
         headers={"Range": "bytes=0-1"})
    show("E. legacy 项目接口", "https://api.gdc.cancer.gov/legacy/api/projects/TCGA-OV")
    show("F. 对照：另一外网端点（判断是否整站 egress 被拦）",
         "https://www.google.com/generate_204")


if __name__ == "__main__":
    main()
