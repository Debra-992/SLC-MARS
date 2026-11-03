# -*- coding: utf-8 -*-
"""GTEx 公共 API 可达性 + 返回粒度实测（经本地 SOCKS 代理）。
若可用，它就能给 OV / SKCM / DLBC 提供"组织匹配的正常参照"（卵巢/皮肤/淋巴），
绕开 GDC 里 normal=0/1/0 的硬缺口。
"""
import json
import requests

P = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}


def hit(url, label):
    try:
        r = requests.get(url, proxies=P, timeout=60, headers={"Accept": "application/json"})
        print(f"[{label}] HTTP {r.status_code}  len={len(r.content)}")
        print("        ", r.text[:400].replace("\n", " "))
        return r
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {e}")
        return None


hit("https://gtexportal.org/API/tissue", "tissue 目录")
hit("https://gtexportal.org/API/expression/median?geneId=SLC36A1,SLC39A9,SLC7A7,SLC25A19"
    "&tissueSiteDetailId=Ovary&dataMediaType=Gene%20Tissue%20Median%20Expression",
    "卵巢 median TPM")
hit("https://gtexportal.org/API/expression/median?geneId=SLC36A1,SLC39A9"
    "&tissueSiteDetailId=Skin%20-%20Sun%20Exposed%20-%20Lower%20Leg&dataMediaType=Gene%20Tissue%20Median%20Expression",
    "皮肤 median TPM")
hit("https://gtexportal.org/API/expression/median?geneId=SLC36A1"
    "&tissueSiteDetailId=Whole%20Blood&dataMediaType=Gene%20Tissue%20Median%20Expression",
    "全血 median TPM")
