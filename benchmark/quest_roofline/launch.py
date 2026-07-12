"""Launch an SGLang server with a process-wide Quest roofline ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from quest_roofline_patch import ENV_NAME, SUPPORTED_MODES, mode_description


def build_environment(
    mode: str, current: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if current is None else current)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    entries = [str(script_dir), str(repo_root / "python")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env[ENV_NAME] = mode
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch python -m sglang.launch_server with a Quest ablation."
    )
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES)
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed verbatim to sglang.launch_server (optionally after --).",
    )
    return parser.parse_args()


def _server_option(server_args: list[str], name: str) -> str | None:
    for index, arg in enumerate(server_args):
        if arg == name:
            if index + 1 >= len(server_args):
                raise ValueError(f"{name} requires a value")
            return server_args[index + 1]
        prefix = f"{name}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def validate_server_args(server_args: list[str]) -> None:
    """Reject launches that cannot exercise the intended Quest FA3 path."""
    if "--enable-hisparse" not in server_args:
        raise ValueError("Quest roofline runs require --enable-hisparse")

    raw_config = _server_option(server_args, "--hisparse-config")
    if raw_config is None:
        raise ValueError("Quest roofline runs require --hisparse-config")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --hisparse-config JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("--hisparse-config must be a JSON object")

    algorithm = config.get("algorithm", "quest")
    if not isinstance(algorithm, str) or algorithm.lower() != "quest":
        raise ValueError("Quest roofline runs require algorithm=quest")
    backend = config.get("backend") or _server_option(
        server_args, "--attention-backend"
    )
    if not isinstance(backend, str) or backend.lower() not in (
        "fa3",
        "flashattention",
    ):
        raise ValueError("Quest roofline runs require the FA3 attention backend")


def main() -> int:
    args = parse_args()
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    if not server_args:
        raise SystemExit("No sglang.launch_server arguments were provided")
    validate_server_args(server_args)

    env = build_environment(args.mode)
    command = [sys.executable, "-m", "sglang.launch_server", *server_args]
    print(f"[quest-roofline] launching {args.mode}: {mode_description(args.mode)}")
    sys.stdout.flush()
    os.execvpe(command[0], command, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
