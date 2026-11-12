# MCPB bundle

Bundle for Smithery's local (stdio) listing and for desktop clients that install `.mcpb` files.
It uses the MCPB `uv` server type: the host installs `rapidbm25` from PyPI with uv, so the
bundle stays small and works on every platform that has a wheel.

Build after the matching rapidbm25 version is on PyPI:

```bash
npx -y @anthropic-ai/mcpb validate mcpb/manifest.json
npx -y @anthropic-ai/mcpb pack mcpb dist/rapidbm25.mcpb
```

Keep `version` here, in `pyproject.toml` (both files) and in `../server.json` in sync.
