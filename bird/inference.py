"""vLLM-backed batch inference.

Designed to live inside a Modal GPU container — `VLLMEngine` is built once per
worker process and reused across many `.chat()` calls. Sampling parameters are
passed per-call so the same engine can serve greedy baseline runs and high-n
self-consistency runs without reloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class GenConfig:
    n: int = 1                 # samples per prompt (>1 enables self-consistency)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)


@dataclass
class GenOutput:
    texts: list[str]           # length == n; one per sample


class VLLMEngine:
    """Thin wrapper around vllm.LLM. Lazy import so this module stays importable on CPU."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        enable_prefix_caching: bool = True,
        download_dir: str | None = None,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        max_num_batched_tokens: int | None = None,
        attention_backend: str | None = None,
    ):
        from vllm import LLM  # noqa: WPS433  (lazy import on purpose)

        self.model = model
        extra_kwargs: dict = {}
        if max_num_batched_tokens is not None:
            # Q3.6 GDN cache alignment requires 2096; vLLM default 8192 silently breaks.
            extra_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if attention_backend is not None:
            # vllm 0.19 takes attention_backend at the LLM ctor; older versions
            # would only honor the env var. We set both for safety.
            extra_kwargs["attention_backend"] = attention_backend
        try:
            self._llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                enable_prefix_caching=enable_prefix_caching,
                download_dir=download_dir,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
                **extra_kwargs,
            )
        except TypeError as e:
            # If `attention_backend` isn't a valid LLM kwarg in this vllm
            # version, fall back to env-var-only mode (already set in image).
            if "attention_backend" in str(e):
                extra_kwargs.pop("attention_backend", None)
                self._llm = LLM(
                    model=model,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=dtype,
                    enable_prefix_caching=enable_prefix_caching,
                    download_dir=download_dir,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager,
                    **extra_kwargs,
                )
            else:
                raise

    def chat(self, conversations: Sequence[list[dict]], cfg: GenConfig) -> list[GenOutput]:
        """Run a batch of chat-message lists through vLLM.

        Returns one `GenOutput` per conversation, each containing `cfg.n` completions
        (vLLM uses the same prompt-cache slot for siblings, so n>1 is cheap).
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
        )
        results = self._llm.chat(messages=list(conversations), sampling_params=sp)
        out: list[GenOutput] = []
        for r in results:
            out.append(GenOutput(texts=[o.text for o in r.outputs]))
        return out

    def complete(self, prompts: Sequence[str], cfg: GenConfig) -> list[GenOutput]:
        """Raw-text completion path — for base models without a chat template.

        Caller renders messages into a single string (see prompts.messages_to_raw_text).
        We bypass vLLM's chat-template logic entirely; the model sees exactly the
        prompt the caller built.
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
        )
        results = self._llm.generate(list(prompts), sampling_params=sp)
        out: list[GenOutput] = []
        for r in results:
            out.append(GenOutput(texts=[o.text for o in r.outputs]))
        return out
