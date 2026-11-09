# -*- coding: utf-8 -*-
"""External comparison baselines (baseline expansion): pure LLM QA / literature RAG / AutoGen / MDAgents.

Together with the stock MetaGPT baseline these form four "no pipeline evidence" controls: they see only the
question stem (question + candidate genes + constraints), never Kira/Cael tool evidence, never CRISPR truth.
Each framework writes answers to outputs/traces/answers_<system>.jsonl, and merge_baselines.py joins them with
"""
