"""LLM backends. Each one implements generate(prompt) -> GenerationResult and
records latency and memory alongside the answer."""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import psutil

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    time_to_first_token_s: Optional[float]
    total_latency_s: float
    tokens_per_second: Optional[float]
    peak_model_mem_mb: float
    extra: dict = field(default_factory=dict)


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UNFINISHED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*(?:should be|is|:)\s*(?P<answer>[^\n.]+)",
    flags=re.IGNORECASE,
)
_REASON_PREFIXES = (
    "let me",
    "okay",
    "ok,",
    "hmm",
    "the question",
    "first,",
    "the user",
    "looking at",
    "the problem",
    "i need to",
    "wait,",
)


def _strip_thinking_trace(text: str) -> str:
    """Pull the answer out of a response that may contain reasoning."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        pass

    text = _THINK_RE.sub("", text)
    text = _UNFINISHED_THINK_RE.sub("", text)
    text = text.strip()
    lowered = text.lower()
    if any(lowered.startswith(p) for p in _REASON_PREFIXES) or len(text.split()) > 40:
        match = _ANSWER_RE.search(text)
        if match:
            return match.group("answer").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines and len(lines[-1].split()) <= 12:
            return lines[-1]
    return text


def _chunk_field(chunk, name: str):
    if isinstance(chunk, dict):
        return chunk.get(name)
    return getattr(chunk, name, None)


def _message_field(message, name: str) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get(name) or ""
    return getattr(message, name, None) or ""


def _ollama_runner_stats() -> Tuple[Set[int], float]:
    """PIDs and total RSS of the Ollama server and its runner processes."""
    pids: Set[int] = set()
    rss_mb = 0.0
    for proc in psutil.process_iter(["name", "memory_info", "pid"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name not in {"ollama.exe", "ollama"} and not name.startswith("llama-server"):
                continue
            pids.add(proc.info["pid"])
            rss_mb += proc.info["memory_info"].rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pids, rss_mb


def _gpu_mem_mb_for_pids(pids: Set[int]) -> float:
    """VRAM used by the given processes, via nvidia-smi. Returns 0 if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0.0

    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            mem = float(parts[-1])
        except ValueError:
            continue
        name = ",".join(parts[1:-1]).lower()
        if pid in pids or "ollama" in name or "llama" in name:
            total += mem
    return total


def _loaded_name_matches(loaded: str, requested: str) -> bool:
    if loaded == requested:
        return True
    return loaded.startswith(requested + ":") or loaded.startswith(requested + "-")


def _ollama_ps_memory(model_name: str) -> tuple[float, float]:
    """Return (size_mb, size_vram_mb) for a loaded model from `ollama ps`."""
    if ollama is None:
        return 0.0, 0.0
    try:
        ps = ollama.ps()
    except Exception:
        return 0.0, 0.0
    models = getattr(ps, "models", None)
    if models is None and isinstance(ps, dict):
        models = ps.get("models") or []
    size_mb = 0.0
    vram_mb = 0.0
    for model in models or []:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
            size = model.get("size")
            vram = model.get("size_vram")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            size = getattr(model, "size", None)
            vram = getattr(model, "size_vram", None)
        if not name or not _loaded_name_matches(str(name), model_name):
            continue
        if size is not None:
            size_mb += float(size) / (1024 * 1024)
        if vram is not None:
            vram_mb += float(vram) / (1024 * 1024)
    return size_mb, vram_mb


def _model_memory(model_name: str) -> dict:
    pids, runner_rss_mb = _ollama_runner_stats()
    ollama_size_mb, ollama_vram_mb = _ollama_ps_memory(model_name)
    smi_gpu_mb = 0.0 if ollama_vram_mb else _gpu_mem_mb_for_pids(pids)
    gpu_mem_mb = smi_gpu_mb or ollama_vram_mb
    if ollama_size_mb:
        peak_model_mem_mb = ollama_size_mb
    elif gpu_mem_mb:
        peak_model_mem_mb = gpu_mem_mb
    else:
        peak_model_mem_mb = 0.0
    return {
        "peak_model_mem_mb": peak_model_mem_mb,
        "peak_gpu_mem_mb": gpu_mem_mb,
        "peak_runner_rss_mb": runner_rss_mb,
        "ollama_ps_size_mb": ollama_size_mb,
    }


class OllamaLLM:
    """A local model served by Ollama. Needs `ollama serve` running and the tag pulled."""

    def __init__(self, model_name: str, options: Optional[dict] = None, timeout_s: float = 180.0):
        if ollama is None:
            raise RuntimeError("ollama package not installed: pip install ollama")
        self.model_name = model_name
        self.options = dict(options or {})
        self.client = ollama.Client(timeout=timeout_s)
        # qwen3 and qwen3.5 support hybrid thinking; force it off for scoring.
        self._is_qwen3 = bool(re.match(r"qwen3(\.\d+)?:", model_name))
        self._cached_mem = None
        if self._is_qwen3:
            # Some tags only honour think= when it's also in the runner options.
            self.options.setdefault("think", False)

    def generate(self, prompt: str, system: Optional[str] = None) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        user_prompt = prompt
        chat_kwargs = {
            "model": self.model_name,
            "options": self.options,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
        }
        if self._is_qwen3:
            user_prompt = (
                "/no_think\n"
                "Return a JSON object with exactly one key named \"answer\". "
                "The answer value must be the short answer only. Do not include reasoning.\n\n"
                + prompt
            )
            chat_kwargs["format"] = "json"

        messages.append({"role": "user", "content": user_prompt})
        chat_kwargs["messages"] = messages

        start = time.perf_counter()
        response = self.client.chat(**chat_kwargs)
        end = time.perf_counter()

        message = _chunk_field(response, "message")
        raw_text = _message_field(message, "content")
        text = _strip_thinking_trace(raw_text)

        if self._cached_mem is None or not self._cached_mem.get("ollama_ps_size_mb"):
            self._cached_mem = _model_memory(self.model_name)
        peak = self._cached_mem

        prompt_eval_ns = _chunk_field(response, "prompt_eval_duration")
        ttft = None
        if prompt_eval_ns is not None:
            try:
                ttft = float(prompt_eval_ns) / 1e9
            except (TypeError, ValueError):
                ttft = None

        eval_count = _chunk_field(response, "eval_count")
        prompt_eval_count = _chunk_field(response, "prompt_eval_count")

        total_latency = end - start

        completion_tokens = None
        if eval_count is not None:
            try:
                completion_tokens = int(eval_count)
            except (TypeError, ValueError):
                completion_tokens = None
        if completion_tokens is None:
            completion_tokens = max(len(text.split()), 1)

        tokens_per_second = (
            completion_tokens / total_latency if total_latency > 0 else None
        )

        prompt_tokens = None
        if prompt_eval_count is not None:
            try:
                prompt_tokens = int(prompt_eval_count)
            except (TypeError, ValueError):
                prompt_tokens = None

        return GenerationResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_to_first_token_s=ttft,
            total_latency_s=total_latency,
            tokens_per_second=tokens_per_second,
            peak_model_mem_mb=peak["peak_model_mem_mb"],
            extra={
                "raw_text": raw_text if raw_text != text else None,
                "peak_gpu_mem_mb": peak["peak_gpu_mem_mb"],
                "peak_runner_rss_mb": peak["peak_runner_rss_mb"],
                "ollama_ps_size_mb": peak["ollama_ps_size_mb"],
            },
        )

    def unload(self) -> None:
        try:
            self.client.generate(model=self.model_name, prompt="", keep_alive=0)
        except Exception:
            pass

    def unload_others(self, keep: set[str]) -> None:
        """Unload any other loaded models so memory readings reflect this one."""
        try:
            ps = ollama.ps()
        except Exception:
            return
        models = getattr(ps, "models", None) or []
        for model in models:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if not name:
                continue
            if any(_loaded_name_matches(str(name), kept) for kept in keep):
                continue
            try:
                self.client.generate(model=name, prompt="", keep_alive=0)
            except Exception:
                continue


def _nvidia_smi_total_used_mb() -> float:
    """System-wide GPU memory in use, summed across cards, via nvidia-smi.

    Backend-agnostic fallback for servers that (unlike Ollama's `ollama ps`)
    expose no per-model memory figure — llama.cpp server, vLLM, LM Studio.
    Returns 0.0 when nvidia-smi is unavailable (Apple Silicon, AMD, CPU-only)
    rather than raising. This is coarser than Ollama's per-model number: it
    reflects everything resident on the GPU during the call, not just the
    model under test, so treat it as a rough signal.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0.0
    total = 0.0
    for line in out.strip().splitlines():
        try:
            total += float(line.strip())
        except ValueError:
            continue
    return total


class OpenAICompatibleLLM:
    """Any server speaking the OpenAI `/v1/chat/completions` API.

    Covers llama.cpp server (default port 8080), LM Studio (1234), vLLM
    (8000), and Ollama's own OpenAI-compatible endpoint. Selected with
    `backend: openai_compatible` plus a `base_url` in the config; switching
    between these tools is then a one-line YAML change.

    `model_name` is passed through to the server as the `model` field —
    llama.cpp server usually ignores it (one model per process), while vLLM
    and LM Studio expect it to match the loaded model's identifier.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "not-needed",
        options: Optional[dict] = None,
        timeout_s: float = 180.0,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed: pip install openai")
        self.model_name = model_name
        self.options = dict(options or {})
        # api_key is required by the client even for local servers that don't
        # check it; "not-needed" is the conventional placeholder in the
        # llama.cpp / vLLM / LM Studio docs.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._cached_gpu_mb = None

    def _params(self) -> dict:
        """Map the benchmark's generic option names onto OpenAI param names."""
        params = dict(self.options)
        num_predict = params.pop("num_predict", None)
        if num_predict is not None:
            params.setdefault("max_tokens", num_predict)
        return params

    def generate(self, prompt: str, system: Optional[str] = None) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        first_token_time = None
        chunks = []
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **self._params(),
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            delta = choices[0].delta.content if choices else None
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(delta)
        end = time.perf_counter()

        raw_text = "".join(chunks)
        text = _strip_thinking_trace(raw_text)

        total_latency = end - start
        ttft = (first_token_time - start) if first_token_time is not None else None
        # No usage numbers over a plain stream, so approximate from words.
        completion_tokens = max(len(raw_text.split()), 1)
        tokens_per_second = (
            completion_tokens / total_latency if total_latency > 0 else None
        )

        # No cross-server "memory used by this model" endpoint exists, so fall
        # back to system-wide GPU memory. Cached after the first successful
        # read so every question in a run doesn't shell out to nvidia-smi.
        if not self._cached_gpu_mb:
            self._cached_gpu_mb = _nvidia_smi_total_used_mb()
        gpu_mb = self._cached_gpu_mb or None

        return GenerationResult(
            text=text,
            model=self.model_name,
            prompt_tokens=None,
            completion_tokens=completion_tokens,
            time_to_first_token_s=ttft,
            total_latency_s=total_latency,
            tokens_per_second=tokens_per_second,
            peak_model_mem_mb=gpu_mb or 0.0,
            extra={
                "raw_text": raw_text if raw_text != text else None,
                "peak_gpu_mem_mb": gpu_mb,
                "peak_runner_rss_mb": None,
                "ollama_ps_size_mb": None,
                # Flag that this memory figure is system-wide, not per-model,
                # so downstream analysis / results tables can caveat it.
                "gpu_mem_is_system_wide": True,
            },
        )


def build_llm(
    model_name: str,
    backend: str = "ollama",
    options: Optional[dict] = None,
    timeout_s: float = 180.0,
    base_url: Optional[str] = None,
    api_key: str = "not-needed",
):
    if backend == "ollama":
        return OllamaLLM(model_name, options=options, timeout_s=timeout_s)
    if backend == "openai_compatible":
        if not base_url:
            raise ValueError(
                f"llm '{model_name}' uses backend: openai_compatible but has no "
                "base_url set in the config"
            )
        return OpenAICompatibleLLM(
            model_name,
            base_url=base_url,
            api_key=api_key,
            options=options,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
