"""Run the existing zero-shot evaluator with a local two-GPU llama.cpp server.

Build a CUDA-enabled llama-server first (see the accompanying Kaggle notebook).
All evaluation flags from scripts.evaluate_gqa_lmstudio are accepted.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from scripts.evaluate_gqa_lmstudio import (
    LMStudioClient, evaluate, image_path, load_examples, parse_args as evaluation_args,
)


DEFAULT_MODEL_DIR = "mistralai/Ministral-3-8B-Instruct-2512-GGUF"
_QUANT = re.compile(r"(?:^|[-_.])(I?Q\d[A-Z0-9_]*|BF16|FP16|F16|FP32|F32)(?=[-.]|$)", re.I)
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def model_identifier(model_dir: str, quantization: str | None = None) -> str:
    """Create a stable alias and output folder from the single model setting."""
    value = str(model_dir).rstrip("/")
    name = re.sub(r"[^a-z0-9_-]+", "-", value.rsplit("/", 1)[-1].lower()).strip("-")
    digest = hashlib.sha256((value + "|" + (quantization or "auto")).encode()).hexdigest()[:10]
    return f"{name or 'gguf-model'}-{digest}"


def quantization_of(name: str) -> str | None:
    match = _QUANT.search(_SHARD.sub(".gguf", Path(name).name))
    return match.group(1).upper() if match else None


def model_family(name: str) -> str:
    stem = Path(_SHARD.sub(".gguf", name)).stem
    stem = _QUANT.sub("", stem)
    stem = re.sub(r"mmproj|projector", "", stem, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def select_gguf_files(names: list[str], quantization: str | None = None) -> tuple[str, str, list[str]]:
    """Select one model family and its projector, including all weight shards."""
    files = sorted(name for name in names if name.lower().endswith(".gguf"))
    projectors = [name for name in files if re.search(r"mmproj|projector", Path(name).name, re.I)]
    weights = [name for name in files if name not in projectors]
    candidates = [name for name in weights if not _SHARD.search(name) or int(_SHARD.search(name).group(1)) == 1]
    if not candidates:
        raise ValueError("MODEL_DIR must contain GGUF model weights (or the first shard).")
    if not projectors:
        raise ValueError("VQA requires a vision GGUF projector named mmproj or projector in MODEL_DIR.")
    if len({model_family(name) for name in candidates}) > 1:
        raise ValueError("MODEL_DIR contains multiple model families. Use a folder or HF repository for one model.")
    requested = quantization.upper() if quantization else None
    if requested:
        candidates = [name for name in candidates if quantization_of(name) == requested]
    else:
        for preferred in ("Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "F16", "FP16", "BF16"):
            matches = [name for name in candidates if quantization_of(name) == preferred]
            if matches:
                candidates = matches
                break
    if len(candidates) != 1:
        raise ValueError(f"Cannot select unique GGUF weights ({requested or 'auto'}): {candidates}. "
                         "Use a directory with one quantization, or append :Q4_K_M to the HF repo ID.")
    model = candidates[0]
    matching = [name for name in projectors if model_family(name) == model_family(model)]
    projectors = matching or [name for name in projectors if not model_family(name)]
    if not projectors:
        raise ValueError("No projector matches the selected model. Keep its matching mmproj in MODEL_DIR.")
    if len(projectors) > 1:
        for preferred in ("F16", "FP16", "BF16", "Q8_0"):
            matches = [name for name in projectors if quantization_of(name) == preferred]
            if matches:
                projectors = matches
                break
    if len(projectors) != 1:
        raise ValueError(f"Multiple matching vision projectors: {projectors}. Use a folder with one projector.")
    shards = [model]
    match = _SHARD.search(model)
    if match:
        shards = [_SHARD.sub(f"-{index:05d}-of-{int(match.group(2)):05d}.gguf", model)
                  for index in range(1, int(match.group(2)) + 1)]
        missing = set(shards) - set(weights)
        if missing:
            raise ValueError(f"Missing GGUF weight shards: {sorted(missing)}")
    return model, projectors[0], shards


def gpu_environment() -> tuple[dict, list[str]]:
    """Respect CUDA visibility and restrict this run to the first two devices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Enable a two-GPU Kaggle accelerator before running.") from error
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = ["0", "1"] if len(devices) >= 2 else []
    else:
        selected = [item.strip() for item in visible.split(",") if item.strip()]
        if any(item == "-1" for item in selected):
            selected = []
        selected = selected[:2]
    if len(devices) < 2 or len(selected) < 2 or len(set(selected)) != 2:
        raise RuntimeError("Two visible NVIDIA GPUs are required; choose GPU T4 x2 in Kaggle.")
    identifiers = {row.split(",", 1)[0].strip() for row in devices}
    for identifier in selected:
        valid_index = identifier.isdecimal() and int(identifier) < len(devices)
        if not valid_index and identifier not in identifiers:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES contains an unavailable GPU: {identifier}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    return environment, devices


def resolve_model(model_dir: str, cache_dir: Path, quantization: str | None = None,
                  revision: str = "main") -> tuple[Path, Path, dict]:
    """Accept a local folder or HF repo; discover weights and vision projector."""
    local = Path(model_dir).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError("MODEL_DIR must be a folder containing GGUF weights and a matching projector.")
        names = [path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()]
        model, projector, shards = select_gguf_files(names, quantization)
        checksums = {}
        for name in [*shards, projector]:
            digest = hashlib.sha256()
            with (local / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[name] = digest.hexdigest()
        return local / model, local / projector, {
            "directory": str(local.resolve()), "model_file": model, "projector_file": projector,
            "quantization": quantization_of(model), "files_sha256": checksums,
        }
    if local.is_absolute() or str(model_dir).startswith((".", "~")):
        raise FileNotFoundError(f"Local MODEL_DIR does not exist: {local}")
    repo = str(model_dir).rstrip("/")
    if repo.startswith("https://huggingface.co/"):
        repo = urlparse(repo).path.strip("/")
    if ":" in repo:
        repo, selected_quant = repo.rsplit(":", 1)
        if quantization and quantization.upper() != selected_quant.upper():
            raise ValueError("Conflicting quantizations in MODEL_DIR and --quantization")
        quantization = selected_quant
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("MODEL_DIR must be a local GGUF folder or Hugging Face owner/model repository.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface-hub before downloading GGUF files.") from error
    api = HfApi()
    resolved = api.model_info(repo, revision=revision).sha
    model, projector, shards = select_gguf_files(api.list_repo_files(repo, revision=resolved), quantization)
    destination = cache_dir / model_identifier(repo) / resolved
    paths = {name: Path(hf_hub_download(
        repo_id=repo, filename=name, revision=resolved, local_dir=str(destination),
    )) for name in [*shards, projector]}
    return paths[model], paths[projector], {
        "repo": repo, "revision": resolved, "quantization": quantization_of(model),
        "model_file": model, "projector_file": projector, "weight_files": shards,
    }


def server_command(binary: Path, model: Path, projector: Path, alias: str,
                   port: int, context_size: int) -> list[str]:
    for artifact in (binary, model, projector):
        if not artifact.is_file():
            raise FileNotFoundError(f"Required file not found: {artifact}")
    return [
        str(binary.resolve()), "--model", str(model.resolve()),
        "--mmproj", str(projector.resolve()), "--alias", alias,
        "--host", "127.0.0.1", "--port", str(port),
        "--n-gpu-layers", "99", "--split-mode", "layer", "--tensor-split", "1,1",
        "--main-gpu", "0", "--ctx-size", str(context_size), "--parallel", "1",
        "--batch-size", "512", "--ubatch-size", "128", "--flash-attn", "off", "--jinja",
    ]


def wait_for_server(process, port: int, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}; inspect its log.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server did not become ready within {timeout:g}s; inspect its log.")


def available_server_port(port: int = 0) -> int:
    """Reuse closing connections; port 0 asks the OS to select a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"Port {port} is occupied. Use --server-port 0 to select a free port."
                ) from error
            raise
        return probe.getsockname()[1]


@contextmanager
def running_server(command: list[str], environment: dict, log_path: Path,
                   port: int, timeout: float):
    # Refuse an occupied port so health checks cannot attach to a different model.
    available_server_port(port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_server(process, port, timeout)
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--llama-server", type=Path,
                        default=Path("/kaggle/working/llama.cpp/build/bin/llama-server"))
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Local GGUF folder or HF repo ID; optional repo:Q8_0 quantization suffix")
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/gguf-models"))
    parser.add_argument("--quantization", help="Optional quantization override; auto prefers Q4_K_M")
    parser.add_argument("--hf-revision", default="main", help="HF commit or revision to download")
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--server-port", type=int, default=0,
                        help="Local port; 0 automatically chooses a free port (default)")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--server-log", type=Path, default=Path("/kaggle/working/llama-server.log"))
    options, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
        evaluation_args(["--help"])
    if options.context_size <= 0 or options.startup_timeout <= 0 or not 0 <= options.server_port <= 65535:
        parser.error("context-size and startup-timeout must be positive; server-port must be 0..65535")
    alias = model_identifier(options.model_dir, options.quantization)
    url = f"http://127.0.0.1:{options.server_port}/v1"
    args = evaluation_args(["--model", alias, "--base-url", url, *remaining])
    if args.base_url.rstrip("/") != url:
        parser.error("The local server URL is managed by --server-port; omit --base-url")
    if "--output-dir" not in remaining and not any(flag.startswith("--output-dir=") for flag in remaining):
        args.output_dir = Path(f"/kaggle/working/results/{alias}_{args.dataset}")
    return options, args


def main(argv=None) -> int:
    options, args = parse_args(argv)
    try:
        environment, devices = gpu_environment()
        if not options.llama_server.is_file():
            raise FileNotFoundError(f"Build CUDA llama-server first: {options.llama_server}")
        # Fail on bad dataset paths before a multi-GB download.
        examples = load_examples(args.questions)
        for example in examples[:args.limit]:
            image_path(args.images, example, args.image_split or args.questions.stem)
        print("NVIDIA GPUs: " + "; ".join(devices), flush=True)
        model, projector, source = resolve_model(options.model_dir, options.cache_dir,
                                               options.quantization, options.hf_revision)
        print(f"Selected weights: {model.name}; vision projector: {projector.name}", flush=True)
        options.server_port = available_server_port(options.server_port)
        args.base_url = f"http://127.0.0.1:{options.server_port}/v1"
        command = server_command(options.llama_server, model, projector, args.model,
                                 options.server_port, options.context_size)
        version = subprocess.run([str(options.llama_server.resolve()), "--version"],
                                 check=True, capture_output=True, text=True)
        args.backend_config = {
            "engine": "llama.cpp", "version": (version.stdout + version.stderr).strip(),
            "model_source": source, "split_mode": "layer", "tensor_split": "1,1",
            "context_size": options.context_size, "parallel": 1, "flash_attention": "off",
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        }
        args.allow_server_port_change = True
        print(f"Starting llama-server on two GPUs at {args.base_url}; log: {options.server_log}", flush=True)
        with running_server(command, environment, options.server_log,
                            options.server_port, options.startup_timeout):
            client = LMStudioClient(args.base_url, args.timeout, args.retries, server_name="llama.cpp")
            if args.model not in client.models():
                raise RuntimeError("llama-server did not expose the configured model alias.")
            if args.list_models:
                print("\n".join(client.models()))
                return 0
            report = evaluate(args, client)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Results: {args.output_dir}", flush=True)
        return 0 if report["complete"] else 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun with identical settings and --resume.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}\nServer log: {options.server_log}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
