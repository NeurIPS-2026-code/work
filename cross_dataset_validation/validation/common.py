from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


VALIDATION_ROOT = Path(__file__).resolve().parents[1]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = _expand(json.load(handle))

    required_sections = {
        "paths",
        "data",
        "worker",
        "evaluator",
        "embedding",
        "experience",
        "hotpotqa",
        "report",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    for key in ("source_root", "locomo_data", "hotpotqa_data", "output_root"):
        raw = Path(config["paths"][key])
        if not raw.is_absolute():
            raw = path.parent / raw
        config["paths"][key] = str(raw.resolve())

    for endpoint_name in ("worker", "evaluator"):
        endpoint = config[endpoint_name]
        if endpoint.get("backend") not in {"vllm", "openai"}:
            raise ValueError(
                f"{endpoint_name}.backend must be 'vllm' or 'openai'"
            )
        for field in ("model", "base_url"):
            if not endpoint.get(field):
                raise ValueError(f"{endpoint_name}.{field} must be set")
    if not config["worker"].get("model_size"):
        raise ValueError("worker.model_size must be set (for example, '3B')")

    low = int(config["experience"]["low_threshold"])
    high = int(config["experience"]["high_threshold"])
    if not (0 <= low < high <= 12):
        raise ValueError(
            "Experience thresholds must satisfy 0 <= low < high <= 12"
        )
    if int(config["experience"].get("retrieval_top_k", 3)) <= 0:
        raise ValueError("experience.retrieval_top_k must be positive")
    start = int(config["hotpotqa"]["start_idx"])
    end = int(config["hotpotqa"]["end_idx"])
    if start < 0 or end <= start:
        raise ValueError("hotpotqa must satisfy 0 <= start_idx < end_idx")

    config["_config_path"] = str(path)
    return config


def source_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["source_root"])


def output_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"])


def bootstrap_source(config: Dict[str, Any]) -> Path:
    root = source_root(config)
    if not root.is_dir():
        raise FileNotFoundError(f"Original source root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def load_source_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any, *, sort_keys: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
        )
        handle.write("\n")
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def endpoint_api_key(endpoint: Dict[str, Any]) -> str:
    env_name = endpoint.get("api_key_env", "")
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value
    return endpoint.get("api_key", "empty") or "empty"


def make_generator(
    config: Dict[str, Any],
    endpoint_name: str,
    *,
    temperature: float,
    max_tokens: int,
    use_schema: bool,
):
    bootstrap_source(config)
    endpoint = config[endpoint_name]
    backend = endpoint.get("backend", "vllm").lower()
    common = {
        "model_name": endpoint["model"],
        "api_key": endpoint_api_key(endpoint),
        "base_url": endpoint.get("base_url"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_schema": use_schema,
        "timeout": endpoint.get("timeout", 180.0),
    }

    if backend == "vllm":
        from gam import VLLMGenerator

        return VLLMGenerator(common)
    if backend == "openai":
        from gam import OpenAIGenerator

        return OpenAIGenerator(common)
    raise ValueError(f"Unsupported backend {backend!r} for {endpoint_name}")


def worker_call_args(config: Dict[str, Any]) -> Dict[str, str]:
    worker = config["worker"]
    backend = worker.get("backend", "vllm").lower()
    return {
        "memory_api_key": endpoint_api_key(worker),
        "memory_base_url": worker["base_url"],
        "memory_model": worker["model"],
        "memory_api_type": backend,
        "research_api_key": endpoint_api_key(worker),
        "research_base_url": worker["base_url"],
        "research_model": worker["model"],
        "research_api_type": backend,
        "working_api_key": endpoint_api_key(worker),
        "working_base_url": worker["base_url"],
        "working_model": worker["model"],
        "working_api_type": backend,
    }


def response_tokens(response: Dict[str, Any]) -> int:
    try:
        usage = response["response"]["usage"]
        if isinstance(usage, dict):
            return int(usage.get("total_tokens", 0) or 0)
        return int(getattr(usage, "total_tokens", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return 0


def response_json(response: Dict[str, Any]) -> Dict[str, Any]:
    parsed = response.get("json")
    if isinstance(parsed, dict):
        return parsed
    text = str(response.get("text", "")).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def embedding_device(config: Dict[str, Any]) -> str:
    requested = str(config["embedding"].get("device", "auto"))
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def environment_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    package_names = [
        "faiss-cpu",
        "FlagEmbedding",
        "numpy",
        "openai",
        "sentence-transformers",
        "torch",
        "transformers",
    ]
    versions: Dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for name in package_names:
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = "not-installed"
    except ImportError:
        pass

    return {
        "experiment_name": config.get("experiment_name"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "worker": {
            key: value
            for key, value in config["worker"].items()
            if key != "api_key"
        },
        "evaluator": {
            key: value
            for key, value in config["evaluator"].items()
            if key != "api_key"
        },
        "embedding": config["embedding"],
    }


def conventional_range_name(start: int, end: int, prefix: str) -> str:
    return f"{prefix}_{start}_{end - 1}.json"


def validate_retriever_artifacts(sample_root: str | Path) -> None:
    root = Path(sample_root)
    required = {
        "page-index snapshot": root / "page_index" / "pages",
        "BM25/Lucene index": root / "bm25_index" / "index",
        "dense embedding matrix": root / "dense_index" / "doc_emb.npy",
    }
    missing = []
    for label, path in required.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
            continue
        if path.is_dir() and not any(path.iterdir()):
            missing.append(f"{label} is empty: {path}")
    if missing:
        raise RuntimeError(
            "A required GAM retriever failed to build. The original runner "
            "catches this error and continues, which would invalidate the "
            "comparison. Missing artifacts:\n" + "\n".join(missing)
        )
