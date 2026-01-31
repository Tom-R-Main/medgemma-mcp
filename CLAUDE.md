# MedGemma MCP Server

## Project Overview
Local-first MCP server wrapping Google's MedGemma 4B-IT model for medical image analysis
and clinical reasoning. Targets the MedGemma Impact Challenge Agentic Workflow Prize.

## Architecture
- `src/medgemma_mcp/server.py` - FastMCP entry point with lifespan model loading
- `src/medgemma_mcp/model/` - Model loading (lifespan) and inference wrapper
- `src/medgemma_mcp/tools/` - MCP tool implementations (4 tools: analyze_image, medical_reason, summarize_fhir, extract)
- `src/medgemma_mcp/prompts/` - Chain-of-thought prompt templates per modality (8 templates)
- `src/medgemma_mcp/preprocessing/` - DICOM→PIL, image loading, FHIR Bundle→text conversion
- `src/medgemma_mcp/safety/` - Confidence extraction and regulatory disclaimers

## Key Technical Decisions
- MedGemma 4B was NOT trained on FHIR data (scores 67.6% vs base Gemma's 70.9% on EHRQA)
- FHIR comprehension must happen in Python preprocessing, not the model
- Chain-of-thought prompting reduces hallucinations by 86.4% - templates are load-bearing
- MedGemma DOES support system role (per official HuggingFace model card)
- Single-turn only - multi-turn is "not evaluated/optimized"
- Each tool call is stateless and self-contained

## Development Rules
- Use `uv` for package management, never pip
- Use `anyio` for async, never `asyncio` directly
- Use pytest with `@pytest.mark.anyio` for async tests
- Use functions for tests, not Test classes
- Type hints required for all code
- Imports at top of file, never inside functions
- Line length: 120 chars max
- Catch specific exceptions, not bare `except Exception:`
- Use `logger.exception("Failed")` not `logger.error(f"Failed: {e}")`

## Running
```bash
uv run medgemma-mcp           # stdio transport (default)
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format .           # format
```

## MCP SDK Patterns
- Import: `from mcp.server.fastmcp import FastMCP, Context`
- Lifespan: `@asynccontextmanager` yielding dataclass, access via `ctx.request_context.lifespan_context`
- Context typing: `ctx: Context[ServerSession, MedGemmaContext]`
- Errors: `from mcp.server.fastmcp.exceptions import ToolError`
- Structured output: return Pydantic BaseModel (auto-detected)
- NOTE: Published `mcp` package uses `FastMCP` in `mcp.server.fastmcp`, NOT `MCPServer` in `mcp.server.mcpserver`
