import atexit
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class TeeStream:
    """Duplicates stream writes to both the original stream and a log file."""

    def __init__(self, original_stream: Any, file_obj: Any) -> None:
        self.original_stream = original_stream
        self.file_obj = file_obj

    def write(self, data: str) -> int:
        res = self.original_stream.write(data)
        if not self.file_obj.closed:
            self.file_obj.write(data)
            self.file_obj.flush()
        return res

    def flush(self) -> None:
        self.original_stream.flush()
        if not self.file_obj.closed:
            self.file_obj.flush()

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()

    def fileno(self) -> Any:
        return getattr(self.original_stream, "fileno", lambda: None)()

    @property
    def encoding(self) -> str:
        return getattr(self.original_stream, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self.original_stream, "errors", "replace")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_stream, name)


class ConsoleLogger:
    """Mirrors stdout, stderr, and loguru logs to console.log in the run directory."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.file: Any | None = None
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.loguru_sink_id: int | None = None
        self._is_active = False
        self._owner_pid: int | None = None

    def start(self) -> "ConsoleLogger":
        if self._is_active:
            return self
        self._owner_pid = os.getpid()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.log_path.open("a", encoding="utf-8")

        try:
            logger.remove()
            logger.add(self.orig_stderr, colorize=True)
            self.loguru_sink_id = logger.add(self.file, colorize=False)
        except Exception:  # noqa: BLE001
            pass

        sys.stdout = TeeStream(self.orig_stdout, self.file)
        sys.stderr = TeeStream(self.orig_stderr, self.file)

        self._is_active = True
        atexit.register(self.close)
        return self

    def close(self) -> None:
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            return
        if not self._is_active:
            return
        self._is_active = False

        if self.loguru_sink_id is not None:
            with contextlib.suppress(ValueError, KeyError, OSError):
                logger.remove(self.loguru_sink_id)
            self.loguru_sink_id = None

        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr

        if self.file is not None and not self.file.closed:
            self.file.flush()
            self.file.close()

    def stop(self) -> None:
        self.close()

    def __enter__(self) -> "ConsoleLogger":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def is_json_serializable_primitive(val: Any) -> bool:
    """Checks whether a value is a JSON-serializable primitive, list, or dict."""
    if isinstance(val, int | float | str | bool | type(None)):
        return True
    if isinstance(val, list | tuple):
        return all(is_json_serializable_primitive(x) for x in val)
    if isinstance(val, dict):
        return all(
            isinstance(k, str) and is_json_serializable_primitive(v)
            for k, v in val.items()
        )
    return False


def extract_dataloader_config(
    loader: Any,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extracts dataloader and dataset parameters into a JSON-serializable dictionary."""
    config: dict[str, Any] = {}

    if loader is not None:
        for attr in ("num_workers", "pin_memory"):
            val = getattr(loader, attr, None)
            if val is not None and is_json_serializable_primitive(val):
                config[attr] = val

        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            for k, v in getattr(dataset, "__dict__", {}).items():
                if k.startswith("_"):
                    continue
                if is_json_serializable_primitive(v):
                    config[k] = v

    if dataloader_kwargs:
        for k, v in dataloader_kwargs.items():
            if is_json_serializable_primitive(v):
                config[k] = v

    return config
