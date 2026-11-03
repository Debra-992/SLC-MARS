# -*- coding: utf-8 -*-
"""B 路线旁路模块：为 DESeq2 面板覆盖不到的题，用【本地已有的 TCGA 肿瘤计数】+
【经 SOCKS 代理补齐的对照样本】跑真·DESeq2，产出与原生 de_workspace_cache 同结构的
扩展面板 de_panel_ext.json，供 topK 全特征臂在 serving 侧注入。

为什么不是"肿瘤 vs 正常"：GDC 里 TCGA-OV 的正常组织表达定量文件=0、TCGA-SKCM=1、
TCGA-DLBC=0；GTEx 公开匿名访问已关闭（API 404 / v8 bucket 403）。所以只能退而求其次：
  SKCM  Metastatic vs Primary Tumor      （肿瘤内对比，GDC 原生分组）
  OV    Recurrent Tumor vs Primary Tumor （肿瘤内对比，正常侧 GDC 一条都没有）
  DLBC  —— GDC 只有 48 个 Primary Tumor 的 GEQ，无任何第二组 → 诚实不产出
关键：产物仍是 DESeq2 的 log2FoldChange/padj，与 v20full 训练时看到的 de_log2fc 同量纲，
不会重演 lever1 的"非 DESeq2 统计量 → z 分数 out-of-distribution"致命坑。

严格隔离：
  * 只写 outputs/slc-side-cache/DEext/ 与 outputs/traces/de_panel_ext.json；
  * 绝不写 GDCdata / de_workspace_cache（那是冠军二分类的 DE 数据源）；
  * 绝不 import decision_action / calib_dump，v20 判定路径零改动。

用法（必须走 mcp-venv，它才有 requests+PySocks+rpy2）：
  . tools\\slc_mars_env.ps1
  outputs\\mcp-venv\\Scripts\\python.exe tools\\_de_ext_build.py            # 全部
  outputs\\mcp-venv\\Scripts\\python.exe tools\\_de_ext_build.py --only SKCM
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = os.environ.get("SLC_ROOT")
if SRC:                                    # multi_agent 源码目录（TCGA.py / DESeq2.py 所在）
    sys.path.insert(0, str(Path(SRC) / "multi_agent"))

PROXY = os.environ.get("SLC_GDC_PROXY", "socks5h://127.0.0.1:1080")
P = {"http": PROXY, "https": PROXY}
S = "https://api.gdc.cancer.gov"
HDR = {"User-Agent": "slc-de-ext/1.0"}
SIDE = ROOT / "outputs" / "slc-side-cache" / "DEext"          # 自己的下载目录，不碰 GDCdata
OUT = ROOT / "outputs" / "traces" / "de_panel_ext.json"
N_PER_GROUP = int(os.environ.get("SLC_DE_EXT_N", "6"))
PADJ = 0.05
TOP_N = int(os.environ.get("SLC_DE_TOPN", "400"))    # 与冠军 serving 的 top_n 一致（面板行数对齐）


def _num(v):
    """NaN -> None：原生面板经 MCP JSON 后缺失 padj 就是 null（实测 BRCA 400 行里 22 行 None）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f

# 对比定义：(癌种标签, 项目, 实验组, 对照组, 对比性质)
CONTRASTS = [
    ("Melanoma", "TCGA-SKCM", "Metastatic", "Primary Tumor", "tumor_internal"),
    ("Ovarian cancer", "TCGA-OV", "Recurrent Tumor", "Primary Tumor", "tumor_internal"),
]


def sample_of(x):
    """GDC /files 展平响应里 case.samples 可能是 list，也可能直接摊成 sample_type 字段。"""
    case = x.get("cases")
    case = case[0] if isinstance(case, list) and case else (case or {})
    ss = case.get("samples")
    if isinstance(ss, list) and ss:
        return ss[0]
    if isinstance(ss, dict):
        return ss
    return case if "sample_type" in case else {}


def gdc_files(project, sample_types):
    body = {"filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": list(sample_types)}},
    ]}, "fields": "file_id,file_name,file_size,cases.samples.sample_type",
        "size": 400, "format": "json"}
    r = requests.post(S + "/files", json=body, proxies=P, headers=HDR, timeout=120)
    r.raise_for_status()
    rows = r.json()["data"]["hits"]
    if isinstance(rows, dict):
        rows = rows.get("hits", [])
    for x in rows:
        x["sample_type"] = sample_of(x).get("sample_type")
    return rows


def local_quant_files(project):
    """原生链路已下载的 STAR 计数文件直接复用（省时间），路径在 GDCdata 下——只读不写。"""
    base = ROOT / "outputs" / "slc-tools-cache" / "GDCdata" / project
    out = []
    if not base.is_dir():
        return out
    mf = base / "manifest.json"
    if mf.is_file():
        data = json.loads(mf.read_text(encoding="utf-8"))
        for f in data.get("files") or []:
            p = base / f["file_name"]
            if p.is_file() and p.stat().st_size > 1_000_000:
                out.append({"file_id": f["file_id"], "file_name": f["file_name"],
                            "sample_type": f.get("sample_type"), "path": p})
    if not out:                                    # 没 manifest 就按名字扫
        for p in base.rglob("*.rna_seq.augmented_star_gene_counts.tsv"):
            if p.stat().st_size > 1_000_000:
                out.append({"file_id": p.name.split(".")[0], "file_name": p.name,
                            "sample_type": None, "path": p})
    return out


def ensure_downloaded(project, recs):
    """把缺的文件下到旁路目录；已存在则跳过（可断点续跑）。"""
    got = []
    for r in recs:
        path = r.get("path") or (SIDE / project / r["file_name"])
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            t0 = time.time()
            with requests.get(f"{S}/data/{r['file_id']}", stream=True, proxies=P,
                              headers=HDR, timeout=(15, 300)) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.rename(path)
            print(f"        ↓ {r['file_name'][:40]} {path.stat().st_size/1e6:.1f}MB "
                  f"{time.time()-t0:.0f}s")
        got.append({**r, "path": path})
    return got


def read_counts(path):
    """与原生 TCGA._counts_from_manifest 逐字同式：去 N_ 前缀行、gene_name 大写为索引、同名求和。
    （不做额外空名过滤——原生不过滤，size factor 才会同分布）"""
    t = pd.read_csv(path, sep="\t", comment="#")
    t = t.loc[~t["gene_name"].astype(str).str.startswith("N_")]
    s = pd.to_numeric(t["unstranded"], errors="raise").astype("int64")
    s.index = t["gene_name"].astype(str).str.upper().loc[s.index]
    return s.groupby(level=0).sum()


def build_matrix(recs):
    series, conds = [], {}
    for r in recs:
        s = read_counts(r["path"])
        s.name = r["file_id"]
        series.append(s)
        conds[r["file_id"]] = r["condition"]
    counts = pd.concat(series, axis=1)
    meta = pd.DataFrame.from_dict(conds, orient="index", columns=["condition"])
    meta.index.name = "sample"
    return counts, meta


def run_deseq2(counts, meta, contrast):
    # 与 multi_agent/r_runtime.prepare_r 同口径：rpy2 内嵌 R 的环境变量必须在
    # import rpy2 之前设好（RPY2_CFFI_MODE=ABI + 屏蔽用户 Renviron/Rprofile），
    # 否则 R 共享库定位/库路径继承与冠军 MCP R 进程不一致。
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
    os.environ.setdefault("R_ENVIRON_USER", os.devnull)
    os.environ.setdefault("R_PROFILE_USER", os.devnull)
    import rpy2.robjects as ro
    # 关键：rpy2 内嵌 R 不读 R_LIBS_USER（实测 .libPaths() 只剩 R 自带库），
    # 必须显式把项目 R-libs 前置进去，否则 importr('DESeq2') 找不到包。
    # 注意 `libPaths` 在 R 里是带点函数 `.libPaths()`，base 命名空间无此导出名，
    # 只能用 ro.r 求值字符串（早先 importr('base').libPaths 会 AttributeError）。
    libs = (os.environ.get("R_LIBS_USER") or "").replace("\\", "/")
    if libs:
        ro.r(f'.libPaths(c("{libs}", .libPaths()))')
    print(f"    .libPaths -> {[str(x) for x in ro.r('.libPaths()')][:3]}")
    from DESeq2 import deseq2_analysis_tool            # 复用 multi_agent 的真·DESeq2
    exp, ctrl = contrast
    counts = counts.loc[counts.mean(axis=1) > 1]       # = 原生 data_processing_tool
    res = deseq2_analysis_tool(counts, meta)
    # —— 以下列名/排序/截断口径逐字对齐原生 tool_operations._de（L90-108），
    #    否则 de_panel_n / de_sig_count / paralog 三个特征与服务所见不同分布。
    res = res.rename(columns={"log2FoldChange": "log2fc", "pvalue": "p_value",
                              "padj": "adj_p_value"})
    res["gene_symbol"] = res["gene"].astype(str).str.upper()   # 原生 counts 行名已是 symbol
    family = res.loc[res["gene_symbol"].str.startswith("SLC")].copy()
    for col in ("log2fc", "p_value", "adj_p_value"):
        family[col] = pd.to_numeric(family[col], errors="coerce")
    family = family.sort_values(["adj_p_value", "p_value"], na_position="last")  # = slc_filter
    significant = int((family["adj_p_value"] < PADJ).sum())     # = 原生 significant_count
    top = [{"gene_symbol": r["gene_symbol"], "gene": str(r["gene"]),
            "log2fc": _num(r["log2fc"]), "p_value": _num(r["p_value"]),
            "adj_p_value": _num(r["adj_p_value"])}
           for _i, r in family.head(TOP_N).iterrows()]
    return top, significant, len(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某 tcga_code，如 SKCM")
    args = ap.parse_args()

    panels, start = [], time.time()
    for label, project, exp, ctrl, kind in CONTRASTS:
        code = project.split("-")[1]
        if args.only and args.only.upper() != code:
            continue
        print(f"\n=== {label} ({project}) {exp} vs {ctrl} [{kind}] ===")
        # 逐组各自查询（GDC 已在服务端按 sample_type 过滤，不再本地二次过滤，避免展平字段歧义）
        by = {}
        for want in (exp, ctrl):
            by[want] = [x for x in gdc_files(project, (want,)) if x.get("file_id")]
            print(f"    GDC {want!r}: {len(by[want])} 个 GEQ 文件")
        locals_ = {l["file_id"]: l for l in local_quant_files(project)}
        groups = {}
        for name, want in (("tumor", exp), ("normal", ctrl)):
            pool = by.get(want, [])
            pool.sort(key=lambda x: (x["file_id"] not in locals_,))   # 本地已有的优先，省下载
            picked = pool[:N_PER_GROUP]
            recs = []
            for x in picked:
                l = locals_.get(x["file_id"])
                recs.append({"file_id": x["file_id"], "file_name": x["file_name"],
                             "condition": name, "path": l["path"] if l else None})
            groups[name] = ensure_downloaded(project, recs)
            print(f"    {name}({want}) 可用={len(pool)} 取用={len(recs)}")
        if any(len(v) < 2 for v in groups.values()):
            print(f"    !! 某组样本 <2，诚实跳过（不伪造 panel）")
            continue
        counts = pd.concat([read_counts(r["path"]).rename(r["file_id"])
                            for g in groups.values() for r in g], axis=1)
        conds = {r["file_id"]: r["condition"] for g in groups.values() for r in g}
        meta = pd.DataFrame.from_dict(conds, orient="index", columns=["condition"])
        meta.index.name = "sample"
        print(f"    矩阵 {counts.shape[0]} 基因 × {counts.shape[1]} 样本，进 DESeq2 ...")
        t0 = time.time()
        try:
            top, sig_n, tested = run_deseq2(counts, meta, (exp, ctrl))
        except Exception as e:
            print(f"    !! DESeq2 失败 {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"    完成 {time.time()-t0:.0f}s | SLC 检测 {tested} 个 | 显著 padj<{PADJ} = {sig_n} "
              f"| 面板行数 = {len(top)}（原生同口径 head({TOP_N})）")
        panels.append({
            "cancer_type": label, "tcga_code": code, "gene_family": "SLC",
            "status": "ok", "significant_count": sig_n, "top_genes": top,
            "method": "DESeq2", "analysis_method": f"DESeq2 ({kind}: {exp} vs {ctrl})",
            "contrast": "tumor/normal", "contrast_kind": kind,
            "contrast_groups": {"tumor": exp, "normal": ctrl},
            "source": f"side-channel GDC-{project} via SOCKS proxy (not GDCdata)",
            "sample_counts": {"tumor": len(groups["tumor"]), "normal": len(groups["normal"])},
            "family_tested_count": tested, "error": None,
            "caveat": ("外部旁路面板：非肿瘤vs正常，de_log2fc 量纲仍是 DESeq2 log2FC，"
                       "但生物学含义为" + ("肿瘤内程序差异" if kind == "tumor_internal" else "肿瘤/血源正常"))
        })

    if panels:
        existing = []
        if OUT.is_file():
            existing = [p for p in json.loads(OUT.read_text(encoding="utf-8"))["panels"]
                        if p["tcga_code"] not in {x["tcga_code"] for x in panels}]
        OUT.write_text(json.dumps(
            {"source": "tools/_de_ext_build.py (side channel, GDC via SOCKS proxy)",
             "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "n_per_group": N_PER_GROUP, "padj": PADJ,
             "panels": existing + panels}, ensure_ascii=False), encoding="utf-8")
        print(f"\n写入 {OUT}  共 {len(existing)+len(panels)} 个面板  总耗时 {time.time()-start:.0f}s")
    else:
        print("\n没有可产出的面板")


if __name__ == "__main__":
    main()
