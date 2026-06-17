"""Stage 4: LLM Layer — 统一接口，支持多后端"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "qwen", "deepseek", "vllm"]


@dataclass
class LLMConfig:
    provider: Provider = "openai"
    model_name: str = "qwen-max"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 512


class LLMClient:
    """统一的 LLM 调用接口"""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self._init_client()

    def _init_client(self):
        cfg = self.config
        api_key = cfg.api_key or os.environ.get("API_KEY", "")

        if cfg.provider in ("openai", "qwen", "deepseek", "vllm"):
            from openai import OpenAI

            base_url_map = {
                "openai": "https://api.openai.com/v1",
                "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "vllm": "http://localhost:8000/v1",
            }
            base_url = cfg.base_url or base_url_map.get(cfg.provider, base_url_map["openai"])
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """生成回答"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def generate_with_logprobs(self, prompt: str, top_n: int = 5) -> list[tuple[str, float]]:
        """获取带概率的生成结果（用于验证）"""
        messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
            logprobs=True,
            top_logprobs=top_n,
        )
        choice = resp.choices[0]
        candidates = []
        if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content[0].top_logprobs:
                candidates.append((token_info.token, token_info.logprob))
        return candidates
