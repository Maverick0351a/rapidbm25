#!/usr/bin/env python3
"""Compare two tool arms on the same code-search questions.

One arm gets the rapidbm25 MCP server. The other gets grep and read_file.
Both arms use the same prompt and the same OpenAI-compatible chat model.
``--dry-run`` swaps in a scripted model so the tools can be checked with no API key.

A live run reads EVAL_MODEL and EVAL_BASE_URL, and the key from EVAL_API_KEY.
EVAL_BASE_URL is the API root that serves ``/chat/completions``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import select
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_STEPS = 8
MAX_TOOL_CHARS = 8000
MAX_READ_LINES = 200
MCP_TIMEOUT_S = 120
HTTP_TIMEOUT_S = 120
SKIP_DIRS = {"__pycache__", "node_modules", "target", "dist", "build", ".venv", "venv"}
DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+(\w+)\b", re.M)
SYSTEM = (
    "Find the single function or class the question asks for. "
    "Use the tools to look at the repository. Do not answer from memory.\n\n"
    "When you know, reply with only this JSON object:\n"
    '{"file": "relative/path.py", "symbol": "ExactName"}\n\n'
    "file is the path relative to the repository root. "
    "symbol is the def or class name in that file."
)
BASELINE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Literal substring search under the repository. Not a regular expression.",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory relative to the repository root. Default is the root."},
                    "glob": {"type": "string", "description": "Filename glob. Default *.py"},
                    "limit": {"type": "integer", "description": "Max matches, 1 to 50. Default 30"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the repository. Lines are numbered. At most 200 lines.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root"},
                    "start": {"type": "integer", "description": "First line, 1-based"},
                    "end": {"type": "integer", "description": "Last line, inclusive"},
                },
            },
        },
    },
]
MCP_TOOLS = {
    "bm25_index_texts",
    "bm25_index_path",
    "bm25_search",
    "bm25_list_indexes",
    "bm25_drop_index",
}
STOP = {
    "which", "function", "returns", "return", "should", "whether", "through",
    "existing", "multiple", "request", "string", "given", "class", "checks",
    "check", "defined", "using", "when", "true", "false", "that", "this",
    "with", "from", "into", "their", "there", "about", "other", "amount",
}


def fail(message: str) -> None:
    raise SystemExit(message)


def norm_file(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content)


def parse_answer(text: str):
    text = content_text(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    file, symbol = obj.get("file"), obj.get("symbol")
    if not isinstance(file, str) or not isinstance(symbol, str):
        return None
    return {"file": norm_file(file), "symbol": symbol.strip()}


def correct_answer(text: str, expected: dict) -> bool:
    got = parse_answer(text)
    if got is None:
        return False
    return got["file"] == norm_file(expected["file"]) and got["symbol"] == expected["symbol"]


def check_grader() -> None:
    gold = {"file": "src/pkg/mod.py", "symbol": "Widget"}
    assert correct_answer('{"file": "src/pkg/mod.py", "symbol": "Widget"}', gold)
    assert correct_answer('```json\n{"file": "./src/pkg/mod.py", "symbol": "Widget"}\n```', gold)
    assert correct_answer('found it\n{"file": "src\\\\pkg\\\\mod.py", "symbol": "Widget"}', gold)
    assert not correct_answer('{"file": "src/pkg/other.py", "symbol": "Widget"}', gold)
    assert not correct_answer('{"file": "src/pkg/mod.py", "symbol": "widget"}', gold)
    assert not correct_answer("I think it is Widget in mod.py", gold)
    assert parse_answer("") is None


def load_tasks(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    url, sha = spec.get("repo_url"), spec.get("sha")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        fail(f"{path} repo_url must be an https://github.com/ URL")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        fail(f"{path} sha must be 40 hex characters")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 20:
        fail(f"{path} needs at least 20 tasks")
    seen = set()
    for task in tasks:
        for key in ("id", "question", "file", "symbol"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                fail(f"task {task.get('id')!r} is missing {key}")
        if task["id"] in seen:
            fail(f"duplicate task id {task['id']}")
        seen.add(task["id"])
        if re.search(rf"\b{re.escape(task['symbol'])}\b", task["question"]):
            fail(f"{task['id']} question contains the answer symbol")
        if norm_file(task["file"]) in task["question"]:
            fail(f"{task['id']} question contains the answer path")
        task["file"] = norm_file(task["file"])
    return spec


def git_env() -> dict:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def run_cmd(cmd: list[str], timeout: int = 180) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=git_env())
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        fail(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc.stdout.strip()


def git(dest: Path, *args: str, timeout: int = 180) -> str:
    # Drop credential helpers so a CI extraheader cannot ride along to GitHub.
    cmd = ["git", "-c", "credential.helper=", "-c", "http.extraheader=", "-C", str(dest), *args]
    return run_cmd(cmd, timeout=timeout)


def ensure_repo(url: str, sha: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").is_dir():
        head = git(dest, "rev-parse", "HEAD")
        if head == sha:
            return dest
    if any(dest.iterdir()):
        # A partial checkout is not reused. The pin is the whole contract.
        import shutil
        shutil.rmtree(dest)
        dest.mkdir(parents=True)
    run_cmd(["git", "-c", "credential.helper=", "init", "-q", str(dest)])
    git(dest, "remote", "add", "origin", url)
    git(dest, "fetch", "--depth", "1", "origin", sha)
    git(dest, "checkout", "--detach", "--quiet", "FETCH_HEAD")
    head = git(dest, "rev-parse", "HEAD")
    if head != sha:
        fail(f"checkout is {head}, pinned sha is {sha}")
    return dest


def definitions(repo: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    root = repo.resolve()
    for dirpath, dirnames, filenames in os_walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = norm_file(str(path.relative_to(root)))
            for match in DEF_RE.finditer(text):
                found.setdefault(match.group(1), []).append(rel)
    return found


def check_gold(repo: Path, tasks: list[dict]) -> None:
    defs = definitions(repo)
    for task in tasks:
        hits = defs.get(task["symbol"], [])
        if hits != [task["file"]]:
            fail(f"{task['id']}: {task['symbol']} is defined at {hits or 'nowhere'}, gold is {task['file']}")


def os_walk(root: Path):
    import os
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        filenames[:] = sorted(filenames)
        yield dirpath, dirnames, filenames


def inside(repo: Path, given: str) -> Path:
    """A path inside ``repo``. Absolute paths are allowed when they stay there.

    ``..`` and symlinks are refused. The prompt tells the model the absolute
    repository root, so grep and read_file have to accept that form.
    """
    if not isinstance(given, str) or not given.strip() or "\x00" in given:
        raise ValueError("path must be a non-empty string")
    root = repo.resolve()
    raw = Path(given)
    if raw.is_absolute():
        cur = Path(raw.anchor)
        parts = raw.parts[1:]
    else:
        cur = root
        parts = raw.parts
    for part in parts:
        if part in ("", ".", ".."):
            raise PermissionError("path is outside the repository")
        cur = cur / part
        if cur.is_symlink():
            raise PermissionError("symlinks are not read")
    if cur.exists():
        resolved = cur.resolve()
        if resolved != root and root not in resolved.parents:
            raise PermissionError("path is outside the repository")
    elif root not in cur.parents and cur != root:
        raise PermissionError("path is outside the repository")
    return cur


def tool_grep(repo: Path, pattern, glob="*.py", limit=30, path=None) -> dict:
    if not isinstance(pattern, str) or not pattern or len(pattern) > 200:
        raise ValueError("pattern must be a non-empty string of at most 200 characters")
    if not isinstance(glob, str) or not glob or "/" in glob or "\\" in glob or len(glob) > 64:
        raise ValueError("glob must be a filename pattern")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50")
    root = repo.resolve()
    base = root if path in (None, "") else inside(repo, path)
    if not base.is_dir():
        raise ValueError("path must be a directory in the repository")
    import fnmatch
    matches = []
    for dirpath, _dirnames, filenames in os_walk(base):
        for name in filenames:
            if not fnmatch.fnmatch(name, glob):
                continue
            file_path = Path(dirpath) / name
            if file_path.is_symlink() or not file_path.is_file():
                continue
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            if size > 2_000_000:
                continue
            raw = file_path.read_bytes()
            if b"\0" in raw[:4096]:
                continue
            rel = norm_file(str(file_path.relative_to(root)))
            for number, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if pattern in line:
                    matches.append({"path": rel, "line": number, "text": line[:200]})
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


def tool_read(repo: Path, path, start=1, end=None) -> dict:
    target = inside(repo, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    if target.stat().st_size > 2_000_000:
        raise ValueError("file is larger than 2 MB")
    raw = target.read_bytes()
    if b"\0" in raw[:4096]:
        raise ValueError("file looks binary")
    lines = raw.decode("utf-8", "replace").splitlines()
    if isinstance(start, bool) or (start is not None and not isinstance(start, int)):
        raise TypeError("start must be an integer")
    if isinstance(end, bool) or (end is not None and not isinstance(end, int)):
        raise TypeError("end must be an integer")
    start = 1 if start is None else start
    if start < 1:
        raise ValueError("start must be >= 1")
    if lines and start > len(lines):
        raise ValueError("start is past the end of the file")
    end = len(lines) if end is None else end
    if end < start:
        raise ValueError("end must be >= start")
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1
    end = min(end, len(lines))
    body = "\n".join(f"{i}|{lines[i - 1]}" for i in range(start, end + 1))
    rel = norm_file(str(target.relative_to(repo.resolve())))
    return {"path": rel, "start": start, "end": end, "text": body}


class Mcp:
    def __init__(self, repo: Path, cache: Path):
        # Resolve before the process starts. A relative root plus cwd set to
        # that same relative path makes Path.resolve() double the prefix, and
        # every bm25_index_path is then outside the allowed roots.
        repo = repo.resolve()
        cache = cache.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
        env.update({
            "RAPIDBM25_ROOTS": str(repo),
            "RAPIDBM25_CACHE": str(cache),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        })
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "rapidbm25.mcp_server"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(repo), env=env,
            )
            self._err: list[str] = []
            threading.Thread(target=self._drain, daemon=True).start()
            self._id = 0
            init = self.rpc("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "rapidbm25-eval", "version": "0"},
            })
            if init.get("result", {}).get("serverInfo", {}).get("name") != "rapidbm25":
                fail("MCP server did not identify itself as rapidbm25")
            self.notify("notifications/initialized")
            listed = self.rpc("tools/list", {})
            tools = listed.get("result", {}).get("tools")
            if not isinstance(tools, list) or {t.get("name") for t in tools} != MCP_TOOLS:
                fail("MCP server did not list the rapidbm25 tools")
            self.tools = tools
        except BaseException:
            self.close()
            raise

    def _drain(self) -> None:
        stderr = self.proc.stderr
        if stderr is None:
            return
        for line in stderr:
            self._err.append(line)
            del self._err[:-20]

    def _read(self) -> str:
        stream = self.proc.stdout
        if stream is None:
            raise RuntimeError("MCP server has no stdout")
        ready, _, _ = select.select([stream], [], [], MCP_TIMEOUT_S)
        if not ready:
            raise TimeoutError("MCP server timed out")
        line = stream.readline()
        if not line:
            tail = "".join(self._err)[-500:]
            raise RuntimeError(f"MCP server closed stdout {tail}".strip())
        return line

    def rpc(self, method: str, params: dict) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        try:
            return json.loads(self._read())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MCP server sent non-JSON: {exc}") from exc

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _send(self, msg: dict) -> None:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError("MCP server is not running")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def call(self, name: str, arguments: dict):
        msg = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message") or "MCP error")
        result = msg.get("result") or {}
        if result.get("isError"):
            content = result.get("content") or []
            detail = content[0].get("text") if content else "tool error"
            raise RuntimeError(detail)
        if "structuredContent" in result:
            return result["structuredContent"]
        content = result.get("content") or []
        return {"text": content[0].get("text") if content else ""}

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def openai_tools(specs: list[dict]) -> list[dict]:
    out = []
    for spec in specs:
        out.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description") or "",
                "parameters": spec.get("inputSchema") or {"type": "object", "properties": {}},
            },
        })
    return out


def shrink(value, limit: int = 500):
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...truncated"
    if isinstance(value, dict):
        return {str(k): shrink(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [shrink(v, limit) for v in value[:20]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:limit]


def preview(value) -> str:
    return json.dumps(value, default=str)[:240]


class OpenAIClient:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key

    def complete(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            fail(f"chat completions returned HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            fail(f"chat completions request failed: {exc.reason}")
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            fail("chat completions response had no choices[0].message")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return message, (usage.get("prompt_tokens"), usage.get("completion_tokens"))


class ScriptedModel:
    """Deterministic stand-in. It only sees the transcript, not the gold label."""

    def __init__(self, arm: str, repo: str, fault: bool):
        self.arm = arm
        self.repo = repo
        self.fault = fault
        self.n = 0
        self.calls = 0

    def complete(self, messages, tools):
        self.n += 1
        if self.fault and self.n == 1:
            if self.arm == "bm25":
                msg = self._call("grep", {"pattern": "not-a-bm25-tool"})
            else:
                msg = self._call("bm25_search", {"index": "repo", "query": "not-a-grep-tool"})
        elif self.fault and self.n == 2:
            if self.arm == "bm25":
                msg = self._call("bm25_search", {"index": "repo", "query": "before-index", "k": 0})
            else:
                msg = self._call("read_file", {"path": "../outside-the-repo"})
        elif self.arm == "bm25":
            msg = self._bm25(messages)
        else:
            msg = self._grep(messages)
        blob = json.dumps({"content": msg.get("content"), "tool_calls": msg.get("tool_calls")}, default=str)
        prompt = max(1, len(json.dumps(messages, default=str)) // 4)
        completion = max(1, len(blob) // 4)
        return msg, (prompt, completion)

    def _call(self, name: str, arguments: dict) -> dict:
        self.calls += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{self.arm}_{self.calls}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }

    def _answer(self, file: str, symbol: str) -> dict:
        return {"role": "assistant", "content": json.dumps({"file": file, "symbol": symbol})}

    def _question(self, messages) -> str:
        for message in messages:
            if message.get("role") == "user":
                text = content_text(message.get("content"))
                return text.split("\n\n", 1)[-1]
        return ""

    def _bm25(self, messages) -> dict:
        indexed = searched = False
        search_body = ""
        for message in messages:
            if message.get("role") != "tool":
                continue
            text = content_text(message.get("content"))
            if text.startswith("error:"):
                continue
            if '"chunks"' in text:
                indexed = True
            if '"results"' in text:
                searched = True
                search_body = text
        if not indexed:
            return self._call("bm25_index_path", {"index": "repo", "path": self.repo, "glob": "*.py"})
        if not searched:
            return self._call("bm25_search", {"index": "repo", "query": self._question(messages), "k": 8})
        return self._answer(*best_hit(self._question(messages), search_blobs(search_body)))

    def _grep(self, messages) -> dict:
        question = self._question(messages)
        grepped = None
        read_body = ""
        for message in messages:
            if message.get("role") != "tool":
                continue
            text = content_text(message.get("content"))
            if text.startswith("error:"):
                continue
            if text.startswith('{"matches"') or '"matches"' in text[:40]:
                grepped = text
            if '"text"' in text and '"start"' in text:
                read_body = text
        if grepped is None:
            return self._call("grep", {"pattern": pick_token(question), "glob": "*.py", "limit": 30})
        if not read_body:
            try:
                payload = json.loads(grepped)
            except json.JSONDecodeError:
                payload = {}
            matches = payload.get("matches") or []
            if not matches:
                return self._call("grep", {"pattern": "def ", "glob": "*.py", "limit": 5})
            hit = matches[0]
            start = max(1, int(hit["line"]) - 30)
            return self._call("read_file", {"path": hit["path"], "start": start, "end": start + 120})
        try:
            payload = json.loads(read_body)
        except json.JSONDecodeError:
            payload = {}
        numbered = payload.get("text") or ""
        plain = "\n".join(re.sub(r"^\d+\|", "", line) for line in numbered.splitlines())
        path = payload.get("path") or ""
        return self._answer(*best_hit(question, [(path, plain)]))


def pick_token(question: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{5,}", question)
    words = [word for word in words if word.lower() not in STOP]
    words.sort(key=len, reverse=True)
    return words[0] if words else "def "


def search_blobs(body: str) -> list[tuple[str, str]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    blobs = []
    for hit in payload.get("results") or []:
        chunk_id = str(hit.get("id") or "")
        path = norm_file(chunk_id.rsplit(":", 1)[0])
        blobs.append((path, str(hit.get("text") or "")))
    return blobs


def best_hit(question: str, blobs: list[tuple[str, str]]) -> tuple[str, str]:
    words = {word.lower() for word in re.findall(r"[A-Za-z]{4,}", question)}
    best = ("", "")
    score = -1
    for path, text in blobs:
        for match in DEF_RE.finditer(text):
            window = text[max(0, match.start() - 80):match.end() + 360].lower()
            got = sum(1 for word in words if word in window)
            if got > score:
                score = got
                best = (path, match.group(1))
    if best[1]:
        return best
    if blobs and blobs[0][0]:
        return blobs[0][0], ""
    return "", ""


def tool_message(call_id: str, text: str) -> dict:
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "\n...truncated"
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def add_tokens(total, extra):
    if total is None or extra is None:
        return None
    return total + extra


def run_task(client, arm: str, task: dict, repo: Path, cache: Path, tools, allowed: set[str], mcp: Mcp | None) -> dict:
    root = str(repo.resolve())
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Repository root: {root}\n\n{task['question']}"},
    ]
    calls = []
    errors = 0
    wrong = 0
    prompt = completion = 0
    answer = ""
    started = time.perf_counter()
    steps = 0
    for _ in range(MAX_STEPS):
        steps += 1
        message, usage = client.complete(messages, tools)
        prompt = add_tokens(prompt, usage[0])
        completion = add_tokens(completion, usage[1])
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            answer = content_text(message.get("content"))
            messages.append({"role": "assistant", "content": answer})
            break
        messages.append({
            "role": "assistant",
            "content": content_text(message.get("content")) or None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            name, raw, call_id, error = decode_call(call)
            record = {"name": name, "arguments": shrink(raw) if isinstance(raw, dict) else shrink(str(raw)), "ok": False}
            if error:
                errors += 1
                record["error"] = error
                record["kind"] = "tool_error"
                calls.append(record)
                messages.append(tool_message(call_id, f"error: {error}"))
                continue
            if name not in allowed:
                wrong += 1
                record["kind"] = "wrong_tool"
                record["error"] = f"unknown tool {name}"
                calls.append(record)
                messages.append(tool_message(call_id, f"error: unknown tool {name}"))
                continue
            try:
                if arm == "bm25":
                    if mcp is None:
                        raise RuntimeError("MCP server is not running")
                    result = mcp.call(name, raw)
                elif name == "grep":
                    result = tool_grep(
                        repo, raw.get("pattern"), raw.get("glob", "*.py"),
                        raw.get("limit", 30), raw.get("path"),
                    )
                elif name == "read_file":
                    result = tool_read(repo, raw.get("path"), raw.get("start", 1), raw.get("end"))
                else:
                    raise RuntimeError(f"unhandled tool {name}")
            except Exception as exc:
                # A rejected tool call is part of this task's record. It must
                # not abort the other arm or the rest of the task list.
                errors += 1
                record["kind"] = "tool_error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                calls.append(record)
                messages.append(tool_message(call_id, f"error: {type(exc).__name__}: {exc}"))
                continue
            record["ok"] = True
            record["kind"] = "ok"
            record["result_preview"] = preview(result)
            calls.append(record)
            messages.append(tool_message(call_id, json.dumps(result)))
    else:
        answer = answer or ""
    return {
        "task_id": task["id"],
        "arm": arm,
        "question": task["question"],
        "expected": {"file": task["file"], "symbol": task["symbol"]},
        "answer": parse_answer(answer),
        "correct": correct_answer(answer, task),
        "tool_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "tool_errors": errors,
        "wrong_tool": wrong,
        "steps": steps,
    }


def decode_call(call: dict):
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return "", {}, str(call.get("id") if isinstance(call, dict) else ""), "tool call is missing a function"
    name = function.get("name")
    call_id = str(call.get("id") or "")
    if not isinstance(name, str) or not name:
        return "", {}, call_id, "tool call is missing a name"
    raw = function.get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return name, {}, call_id, f"invalid JSON arguments: {exc}"
    if not isinstance(raw, dict):
        return name, {}, call_id, "tool arguments must be an object"
    return name, raw, call_id, None


def median(values):
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return statistics.median(nums)


def arm_summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": n,
        "correct": correct,
        "correctness": (correct / n) if n else None,
        "median_tool_calls": median([len(row["tool_calls"]) for row in rows]),
        "median_prompt_tokens": median([row["prompt_tokens"] for row in rows]),
        "median_completion_tokens": median([row["completion_tokens"] for row in rows]),
        "median_wall_time_s": median([row["wall_time_s"] for row in rows]),
        "median_tool_errors": median([row["tool_errors"] for row in rows]),
        "total_tool_errors": sum(row["tool_errors"] for row in rows),
        "median_wrong_tool": median([row["wrong_tool"] for row in rows]),
        "total_wrong_tool": sum(row["wrong_tool"] for row in rows),
    }


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e9:
            return str(int(value))
        return f"{value:.3f}"
    return str(value)


def write_summary(path: Path, report: dict) -> None:
    arms = report["arms"]
    lines = [
        "# Agent eval",
        "",
        f"Repository: `{report['repo_url']}` @ `{report['sha']}`",
        f"Model: `{report['model']}`",
        "Same prompt for both arms. `bm25` is the rapidbm25 MCP server. `grep` is literal search plus read_file.",
        "",
        "| arm | n | correct | correctness | median tool calls | median prompt tokens | median completion tokens | median wall s | median tool errors (total) | median wrong-tool (total) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("bm25", "grep"):
        arm = arms[name]
        rate = "n/a" if arm["correctness"] is None else f"{arm['correctness']:.3f}"
        lines.append(
            "| {arm} | {n} | {correct} | {rate} | {calls} | {prompt} | {completion} | {wall} | {errors} ({etotal}) | {wrong} ({wtotal}) |".format(
                arm=name,
                n=arm["n"],
                correct=arm["correct"],
                rate=rate,
                calls=fmt(arm["median_tool_calls"]),
                prompt=fmt(arm["median_prompt_tokens"]),
                completion=fmt(arm["median_completion_tokens"]),
                wall=fmt(arm["median_wall_time_s"]),
                errors=fmt(arm["median_tool_errors"]),
                etotal=arm["total_tool_errors"],
                wrong=fmt(arm["median_wrong_tool"]),
                wtotal=arm["total_wrong_tool"],
            )
        )
    bm25 = arms["bm25"]["correctness"]
    grep = arms["grep"]["correctness"]
    delta = None if bm25 is None or grep is None else bm25 - grep
    lines.append("")
    lines.append(f"bm25 correctness minus grep correctness: {fmt(delta)}")
    lines.append("")
    lines.append("| task | bm25 | grep |")
    lines.append("|---|---|---|")
    by_id: dict[str, dict[str, bool]] = {}
    order = []
    for row in report["runs"]:
        if row["task_id"] not in by_id:
            order.append(row["task_id"])
            by_id[row["task_id"]] = {}
        by_id[row["task_id"]][row["arm"]] = row["correct"]
    for task_id in order:
        marks = by_id[task_id]
        lines.append(
            f"| {task_id} | {'yes' if marks.get('bm25') else 'no'} | {'yes' if marks.get('grep') else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def public_base_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.username or parts.password:
        fail("EVAL_BASE_URL must not include a username or password. Put the key in EVAL_API_KEY.")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def live_settings() -> tuple[str, str, str]:
    missing = [name for name in ("EVAL_API_KEY", "EVAL_MODEL", "EVAL_BASE_URL") if not os.environ.get(name, "").strip()]
    if missing:
        fail(
            "Agent eval did not run. Missing: " + ", ".join(missing) + ". "
            "Set EVAL_API_KEY, EVAL_MODEL, and EVAL_BASE_URL. "
            "EVAL_BASE_URL is the OpenAI-compatible root that serves /chat/completions."
        )
    base = os.environ["EVAL_BASE_URL"].strip()
    parts = urllib.parse.urlsplit(base)
    host = (parts.hostname or "").lower()
    local_http = parts.scheme == "http" and host in {"localhost", "127.0.0.1"}
    if parts.scheme != "https" and not local_http:
        fail("EVAL_BASE_URL must be https, or http://localhost or http://127.0.0.1")
    return os.environ["EVAL_MODEL"].strip(), base, os.environ["EVAL_API_KEY"]


def check_dry_run(report: dict) -> None:
    runs = report["runs"]
    tasks = {row["task_id"] for row in runs}
    if len(tasks) < 20 or len(runs) != len(tasks) * 2:
        fail(f"dry-run wrote {len(runs)} rows for {len(tasks)} tasks")
    for arm, need in (("bm25", {"bm25_index_path", "bm25_search"}), ("grep", {"grep", "read_file"})):
        rows = [row for row in runs if row["arm"] == arm]
        if not rows:
            fail(f"dry-run missing {arm}")
        if sum(row["wrong_tool"] for row in rows) < 1:
            fail(f"dry-run {arm} did not record a wrong-tool choice")
        if sum(row["tool_errors"] for row in rows) < 1:
            fail(f"dry-run {arm} did not record a tool error")
        for row in rows:
            ok = {call["name"] for call in row["tool_calls"] if call.get("ok")}
            if not need <= ok:
                missed = ", ".join(sorted(need - ok))
                errors = "; ".join(
                    f"{call.get('name')}: {call.get('error')}"
                    for call in row["tool_calls"]
                    if call.get("error")
                )
                fail(f"dry-run {row['task_id']} {arm} did not succeed at {missed}" + (f" ({errors})" if errors else ""))
            for key in ("correct", "tool_calls", "prompt_tokens", "completion_tokens", "wall_time_s", "tool_errors", "wrong_tool"):
                if key not in row:
                    fail(f"dry-run row missing {key}")
    text = json.dumps(report)
    secret = os.environ.get("EVAL_API_KEY")
    if secret and secret in text:
        fail("dry-run report contains EVAL_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rapidbm25 MCP search with a grep baseline.")
    parser.add_argument("--dry-run", action="store_true", help="use the scripted model; no API key")
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks.json")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    check_grader()
    spec = load_tasks(args.tasks)
    args.out.mkdir(parents=True, exist_ok=True)
    repo = ensure_repo(spec["repo_url"], spec["sha"], args.out / "repo")
    check_gold(repo, spec["tasks"])
    cache = args.out / "cache"
    if args.dry_run:
        model, base_url = "scripted", None
        clients = None
    else:
        model_name, base, key = live_settings()
        model, base_url = model_name, public_base_url(base)
        client = OpenAIClient(model_name, base, key)
        clients = {"bm25": client, "grep": client}
    runs = []
    mcp_tools = None
    for index, task in enumerate(spec["tasks"]):
        for arm in ("bm25", "grep"):
            mcp = None
            try:
                if arm == "bm25":
                    mcp = Mcp(repo, cache)
                    tools = openai_tools(mcp.tools)
                    allowed = {tool["name"] for tool in mcp.tools}
                    if mcp_tools is None:
                        mcp_tools = [tool["name"] for tool in mcp.tools]
                else:
                    tools = BASELINE_TOOLS
                    allowed = {"grep", "read_file"}
                if args.dry_run:
                    client = ScriptedModel(arm, str(repo.resolve()), fault=(index == 0))
                else:
                    client = clients[arm]
                runs.append(run_task(client, arm, task, repo, cache, tools, allowed, mcp))
            finally:
                if mcp is not None:
                    mcp.close()
    report = {
        "repo_url": spec["repo_url"],
        "sha": spec["sha"],
        "model": model,
        "base_url": base_url,
        "dry_run": bool(args.dry_run),
        "prompt": SYSTEM,
        "mcp_tools": mcp_tools,
        "arms": {
            "bm25": arm_summary([row for row in runs if row["arm"] == "bm25"]),
            "grep": arm_summary([row for row in runs if row["arm"] == "grep"]),
        },
        "runs": runs,
    }
    secret = os.environ.get("EVAL_API_KEY")
    encoded = json.dumps(report, indent=2)
    if secret and secret in encoded:
        fail("refusing to write results that contain EVAL_API_KEY")
    out_json = args.out / "runs.json"
    out_md = args.out / "summary.md"
    out_json.write_text(encoded + "\n", encoding="utf-8")
    write_summary(out_md, report)
    if args.dry_run:
        check_dry_run(report)
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
