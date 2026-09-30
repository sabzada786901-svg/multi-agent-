# LangGraph Multi-Agent AI Assistant (Windows + VS Code, free-tier first)

Strict **PDF RAG** + **GitHub MCP** + **Google Calendar MCP** + **Gmail MCP**, coordinated by a LangGraph orchestrator with input/output guardrails and human confirmation for every side effect.

> Verification status: free-tier facts and package names below were checked against public docs on **28 Sep 2026**. The code was written and syntax-checked, and the tests are offline (fake LLM/embeddings), but **I could not run `pip install` or the test-suite in the build sandbox (no network)**. Run `pytest` first on your machine; if a dependency version differs, fix it per *Troubleshooting*.

## 1. Overview & 2. Architecture

```
USER -> INPUT GUARDRAIL -> ORCHESTRATOR -> rag_agent | github_agent | calendar_agent | email_agent
                                              |            \___ MCP (stdio/HTTP) ___/
                                        Qdrant Cloud (free)
every agent result -> CONFIRM (human-in-the-loop) -> next step ... -> OUTPUT GUARDRAIL -> USER
```

## 3. LangGraph flow

`input_guardrail` -> (greeting/unrelated/unsafe: direct reply) or `orchestrator` -> one node per plan step -> `confirm` (uses `interrupt()`; pauses until you answer yes/no) -> next step -> `finalize` -> `output_guardrail`.
State (`app/state.py`) carries `plan`, `step`, `agent_outputs`, `retrieved_context`, `pending_action`, etc., so later agents see earlier agents' results (e.g. Calendar -> Email).
Multi-step example: *"Schedule a meeting with John tomorrow at 3 PM and email him"* -> Calendar (checks conflicts, proposes event) -> you confirm -> Email (drafts, proposes send) -> you confirm.

## 4. RAG architecture (strict, PDF-only)

PDF -> `pypdf` (per page) -> chunks (1000 chars / 150 overlap, keeps `document_name`, `page_number`, `chunk_id`) -> **local** MiniLM embeddings -> Qdrant Cloud.
Answering: retrieve top-K above `MIN_RETRIEVAL_SCORE`. **No hits -> the LLM is never called** ("I couldn't find this information in the uploaded PDF."). With hits, the LLM sees only `<document_data>` blocks and a system prompt that forbids outside knowledge, then `check_grounding` validates the answer; unsupported answers are replaced by the fixed fallback. Source lines (`file.pdf — Page N`) are built **by code from metadata**, never by the model.

## 5. Guardrails

| Guardrail | What it does |
|---|---|
| Input | length/control-char validation, prompt-injection patterns, greeting shortcut (English + Roman Urdu, no LLM call), structured classification into GREETING / PDF_RELATED / GITHUB / CALENDAR / EMAIL / MULTI_AGENT / UNRELATED / UNSAFE (keyword fallback if the LLM is down) |
| Grounding | `GroundingCheck{grounded, supported, reason}` via LLM + lexical-overlap safety net; if the validator is down, a strict lexical check is used (fails closed) |
| Output | enforces failed grounding, blocks system-prompt leaks, redacts API keys/tokens |
| PDF injection | retrieved text is wrapped as DATA, flagged in logs, and the answer is still grounding-checked and secret-redacted |
| Tool safety | tool results (emails, issues, event titles) are treated as untrusted; any write-verb tool call is queued for confirmation; e-mail addresses are validated before queuing |

## 6. Free-tier services (verified 28 Sep 2026; limits change — re-check)

| Service | Free tier | Account / key | May become paid |
|---|---|---|---|
| **Qdrant Cloud** (chosen vector DB) | 1 node, 0.5 vCPU, 1 GB RAM, 4 GB disk (~1M 768-dim vectors); no credit card. **Inactive free clusters are suspended after 1 week and deleted after 4 weeks.** | Qdrant Cloud account -> cluster URL + API key | Larger/HA clusters (Standard from ~$25/mo) |
| **OpenRouter** | Models whose id ends in `:free`; no card. Sources disagree on quotas (about 20 req/min; roughly 50-200 req/day without purchased credits) | openrouter.ai key (you have one) | Non-free models; buying credits raises daily limits |
| **Groq** | Free dev tier, no card. Reported (third-party summary of Groq docs, Aug 2026): ~30 req/min; `llama-3.3-70b-versatile` ~1,000 req & 100K tokens/day; `llama-3.1-8b-instant` ~14,400 req & 500K tokens/day. Limits are per organization | console.groq.com key (you have one) | Developer tier with a card for ~10x limits |
| sentence-transformers | Fully local, free | none | never |
| GitHub MCP (official remote server) | Free with any GitHub account | fine-grained PAT | Copilot features are separate |
| Google Calendar + Gmail APIs | Free within normal quotas | Google Cloud project + OAuth client | n/a for personal use |

**Important:** one PDF question costs about 3 LLM calls (classify, answer, grounding check); MCP tasks cost several more. On OpenRouter's free quota that is only a handful of questions per day. If you hit 429s, set `LLM_PROVIDER=groq` (it has far higher daily limits) — with both keys present the other provider is used automatically as fallback.

## 7. OpenRouter setup / 8. Groq setup

Put keys in `.env`. `OPENROUTER_MODEL` defaults to `openai/gpt-oss-120b:free` (appeared in the free list checked on 28 Sep 2026; needs tool-calling support). If it disappears, pick another `:free` model with tool support at https://openrouter.ai/models?max_price=0 and change `OPENROUTER_MODEL`. Groq: `GROQ_MODEL=llama-3.3-70b-versatile` (see https://console.groq.com/docs/models).

## 9. Local embeddings

`sentence-transformers/all-MiniLM-L6-v2` (384-dim). First use downloads ~90 MB to `%USERPROFILE%\.cache\huggingface`. Runs on CPU (~300-500 MB RAM); GPU not needed. `pip install sentence-transformers` also installs PyTorch (~200 MB CPU wheel on Windows). Change with `EMBEDDING_MODEL=` (then re-ingest; the collection dimension must match — delete the collection in the Qdrant console first).

## 10. Hosted vector DB setup (Qdrant Cloud)

1. https://cloud.qdrant.io -> sign up (no card) -> **Create cluster** -> Free tier.
2. Copy the cluster URL (`https://xxxx.cloud.qdrant.io:6333`) and create an API key -> `VECTOR_DB_URL`, `VECTOR_DB_API_KEY` in `.env`.
3. Keep it alive: use the assistant at least weekly, otherwise the cluster is suspended (reactivate in the console) and later deleted.

## 11. GitHub MCP setup

Uses GitHub's official remote MCP server (`https://api.githubcopilot.com/mcp/`) over streamable HTTP with a PAT in the `Authorization` header (no Docker needed).
GitHub -> Settings -> Developer settings -> Fine-grained tokens -> select repos; grant read access to Contents, Issues, Pull requests (add Issues: write only if you want issue creation). Put it in `GITHUB_TOKEN`. The agent exposes at most 30 relevant tools; anything that creates/changes data needs your confirmation.

## 12. Google Calendar MCP / 13. Email MCP / 14. OAuth setup

Both need **Node.js 18+** (`npx`) — https://nodejs.org. In PowerShell, use the `cmd.exe /c` wrapper below because `npx` may resolve to the blocked `npx.ps1` script.
1. Google Cloud Console -> new project -> enable **Google Calendar API** and **Gmail API** -> OAuth consent screen (External, add your account as a *test user*) -> Credentials -> **OAuth client ID -> Desktop app** -> download JSON.
2. **Calendar** (`@cocal/google-calendar-mcp`): save the JSON somewhere private and set `GOOGLE_OAUTH_CREDENTIALS` in `.env` to its full path. In PowerShell, authenticate once:
   ```powershell
   $env:GOOGLE_OAUTH_CREDENTIALS = "C:\path\gcp-oauth.keys.json"
   cmd.exe /c "npx.cmd -y @cocal/google-calendar-mcp auth"
   ```
   Complete the browser consent flow. Keep this JSON outside the repo; the server stores its own token separately.
3. **Gmail** (`@gongrzhe/server-gmail-autoauth-mcp`):
   ```powershell
   mkdir $env:USERPROFILE\.gmail-mcp
   copy gcp-oauth.keys.json $env:USERPROFILE\.gmail-mcp\
   cmd.exe /c "npx.cmd -y @gongrzhe/server-gmail-autoauth-mcp auth"
   ```
   Tokens are saved in `%USERPROFILE%\.gmail-mcp\credentials.json`.
4. If you prefer other MCP servers, override `CALENDAR_MCP_ARGS` / `EMAIL_MCP_ARGS` in `.env` (tool names are discovered at runtime; only explicitly allowlisted tools are exposed, and write-verb tools are confirmation-gated).

Limitations: the Gmail server has **no scheduled sending** — the agent says so. Google apps in "Testing" mode issue refresh tokens that expire after 7 days; just re-run the auth step.

## 15. Windows installation / 16. VS Code

```powershell
cd multi-agent-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1          # CMD: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env               # then fill in the keys
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In VS Code: *Python: Select Interpreter* -> `.venv`. `.vscode/` is git-ignored.

## 17. PDF ingestion / 18. Running / 19. Testing

```powershell
python -m app.ingest "data\pdfs\document.pdf"     # or: python -m app.ingest   (indexes data\pdfs\*.pdf)
python -m app.main
pytest -q
```
Re-ingesting a file replaces its old chunks. Scanned/image-only PDFs have no text (no OCR included) and are rejected with a clear message.
Tests (offline, no keys): greeting, greeting+PDF routing, PDF answer with source, missing info, out-of-scope, PDF prompt injection, hallucination rejection, confirmation gating for calendar/email/GitHub, timezone `Asia/Karachi` (+05:00).

## 20. Example prompts

`Hello` · `Kaisa ho?` · `Hi, what does the PDF say about authentication?` · `What is the capital of France?` (refused) · `Show my open issues on GitHub` · `What meetings do I have tomorrow?` · `Create a meeting tomorrow at 3 PM` · `Draft an email to john@example.com about the project` · `Read the requirements from the PDF, find the related repo, and draft an email summarizing it`.

## 21. Troubleshooting

| Problem | Fix |
|---|---|
| `429` / rate limit message | free quota used; wait, or `LLM_PROVIDER=groq`, or another `:free` model |
| Model not found / no tool support | change `OPENROUTER_MODEL` (needs tool calling for MCP agents) |
| "vector database is unreachable" | free cluster suspended -> reactivate in Qdrant console; check URL/key |
| `npx not found` | install Node.js, reopen the terminal |
| Calendar/Gmail "not authorised" | redo the OAuth step; re-run after 7 days if app is in Testing mode |
| Invalid TIMEZONE | `pip install tzdata` |
| `ImportError` from langchain/langgraph | `pip install -U -r requirements.txt`; check `logs\app.log` |

Full details of every failure go to `logs\app.log` (secrets redacted); the console only shows friendly messages.

## 22. Security

Secrets only in `.env` (git-ignored) · no secrets in code or logs (redaction filter) · PDF/e-mail/issue text = untrusted data · every create/update/delete/send needs an explicit *yes* · address validation · least-privilege PAT and tool allow-lists (delete/merge/push tools are not exposed) · output redaction and system-prompt-leak block.

## 23. Free-tier limitations

Qdrant free cluster auto-suspends; OpenRouter/Groq daily quotas are small and can change; free models vary in tool-calling quality (if the agent misuses tools, try another model); MCP servers are third-party community projects except GitHub's.

## 24. Future improvements

OCR for scanned PDFs · hybrid (BM25 + vector) retrieval and reranking · persistent checkpointer (SQLite) so confirmations survive restarts · a web UI · streaming responses · evaluation set for grounding.
