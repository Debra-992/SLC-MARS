# -*- coding: utf-8 -*-
"""probe：单题跑一次 autogen 双代理对话，打印原始消息，诊断 JSON 解析失败原因。"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    import autogen
    from knowledge_agents.baselines.common import build_prompt, llm_settings, load_inputs

    cfg = llm_settings()
    rec = load_inputs()[0]
    llm_config = {
        "config_list": [{"model": cfg["model"], "api_key": cfg["api_key"], "base_url": cfg["base_url"]}],
        "timeout": 90,
    }
    assistant = autogen.AssistantAgent(
        name="assistant", llm_config=llm_config,
        system_message="You are a helpful biomedical assistant. 回答完后单独一行输出 TERMINATE。")
    proxy = autogen.UserProxyAgent(name="proxy", human_input_mode="NEVER", code_execution_config=False)
    proxy.initiate_chat(assistant, message=build_prompt(rec), clear_history=True, max_turns=2)
    hist = assistant.chat_messages[proxy]
    print("HISTORY LENGTH:", len(hist))
    for i, m in enumerate(hist):
        print(f"--- msg {i} keys={list(m.keys())} role={m.get('role')}")
        print(str(m.get("content"))[:300])


if __name__ == "__main__":
    main()
