# -*- coding: utf-8 -*-
"""外部对比 baseline 集合（baseline 扩展）：纯LLM问答 / 文献RAG / AutoGen / MDAgents。

与 原生MetaGPT baseline 一起构成 4 个"无管线证据"对照：只拿题面(问题+候选基因+约束)，
不接 Kira/Cael 工具证据、不碰 CRISPR 真值。各框架答案落 outputs/traces/answers_<system>.jsonl，
由 merge_baselines.py 与管线两臂(heuristic/auto)+原生MetaGPT 合并成同表公平对比。
"""
