from __future__ import annotations

import os
import time
from pathlib import Path
from string import Template
from typing import Any

import requests

from systems.base import AskResult, BaseSystem


def _deep_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


class HttpJsonSystem(BaseSystem):
    """
    Config-driven HTTP client for proprietary APIs (e.g. SentrySearch or an internal service).
    Tune paths and JSON field names in config/systems.yaml under this system's params.
    """

    def __init__(
        self,
        name: str = "http_json",
        base_url_env: str = "",
        default_base_url: str = "http://127.0.0.1:8000",
        upload: dict[str, Any] | None = None,
        ask: dict[str, Any] | None = None,
        env_chat_id: str = "",
        default_chat_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._base_url_env = base_url_env
        self._default_base_url = default_base_url
        self._upload_cfg = upload or {}
        self._ask_cfg = ask or {}
        self._env_chat_id = env_chat_id
        self._default_chat_id = default_chat_id
        self._session = requests.Session()
        self._remote_by_video: dict[str, str] = {}

    def _base_url(self) -> str:
        if self._base_url_env:
            v = os.environ.get(self._base_url_env)
            if v:
                return v.rstrip("/")
        return self._default_base_url.rstrip("/")

    def _substitute(self, template_val: Any, mapping: dict[str, str]) -> Any:
        if isinstance(template_val, str):
            return Template(template_val).safe_substitute(mapping)
        if isinstance(template_val, dict):
            return {k: self._substitute(v, mapping) for k, v in template_val.items()}
        if isinstance(template_val, list):
            return [self._substitute(x, mapping) for x in template_val]
        return template_val

    def upload_video(self, video_path: Path, video_id: str) -> str:
        cfg = self._upload_cfg
        method = (cfg.get("method") or "POST").upper()
        path = cfg.get("path") or "/upload"
        url = self._base_url() + path
        file_field = cfg.get("file_field") or "file"
        chat_id = os.environ.get(self._env_chat_id) if self._env_chat_id else None
        chat_id = chat_id or self._default_chat_id
        mapping = {"video_id": video_id, "chat_id": chat_id or ""}
        extra = self._substitute(cfg.get("extra_form_fields") or {}, mapping)
        path_resolved = video_path.resolve()
        if not path_resolved.is_file():
            raise FileNotFoundError(path_resolved)
        data = {k: str(v) for k, v in extra.items()} if extra else None
        with open(path_resolved, "rb") as f:
            files = {file_field: (path_resolved.name, f, "application/octet-stream")}
            r = self._session.request(method, url, files=files, data=data, timeout=600)
        r.raise_for_status()
        try:
            body = r.json()
        except Exception:
            body = {"_raw_text": r.text}
        id_path = cfg.get("response_id_path") or "id"
        rid = _deep_get(body, id_path) if "." in id_path or isinstance(body, dict) else body.get(id_path) if isinstance(body, dict) else None
        if rid is None and isinstance(body, dict):
            rid = body.get("runId") or body.get("videoId") or body.get("id")
        if rid is None:
            raise RuntimeError(f"Could not parse remote id from upload response (path {id_path!r}): {body}")
        rid_str = str(rid)
        self._remote_by_video[video_id] = rid_str
        return rid_str

    def ask(self, remote_video_id: str, question: str, video_id: str) -> AskResult:
        cfg = self._ask_cfg
        method = (cfg.get("method") or "POST").upper()
        path = cfg.get("path") or "/ask"
        url = self._base_url() + path
        chat_id = os.environ.get(self._env_chat_id) if self._env_chat_id else None
        chat_id = chat_id or self._default_chat_id
        mapping = {
            "video_id": video_id,
            "remote_video_id": remote_video_id,
            "question": question,
            "chat_id": chat_id or "",
        }
        t0 = time.perf_counter()
        if cfg.get("json_body_template"):
            payload = self._substitute(cfg["json_body_template"], mapping)
            r = self._session.request(
                method,
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
        elif cfg.get("query_template"):
            params = self._substitute(cfg["query_template"], mapping)
            r = self._session.request(method, url, params=params, timeout=300)
        else:
            r = self._session.request(
                method,
                url,
                json={"question": question, "videoId": remote_video_id},
                timeout=300,
            )
        r.raise_for_status()
        try:
            body = r.json()
        except Exception:
            body = {"_raw_text": r.text}
        text_path = cfg.get("response_text_path") or "answer"
        text = _deep_get(body, text_path) if "." in text_path else body.get(text_path) if isinstance(body, dict) else None
        if text is None:
            text = body.get("response") or body.get("text") or str(body)
        ranked = None
        rp = cfg.get("response_ranked_ids_path")
        if rp and isinstance(body, dict):
            raw_list = _deep_get(body, rp)
            if isinstance(raw_list, list):
                ranked = [str(x) for x in raw_list]
        return AskResult(
            text=str(text).strip(),
            latency_sec=time.perf_counter() - t0,
            raw=body if isinstance(body, dict) else {"_body": body},
            ranked_segment_ids=ranked,
            metadata={},
        )
