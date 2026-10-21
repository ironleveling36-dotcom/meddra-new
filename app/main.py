"""
MedDRA Coding Assistant — FastAPI app.

Fixes vs v2.0:
  ✓ get_engine() is never called synchronously on the hot path during cold start.
    Requests that arrive before the engine is warm get a clean 503, not a hung event loop.
  ✓ /health returns engine status without blocking.
  ✓ All other logic identical to v2.0.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import engine as engine_mod
from app.config import settings
from app.engine import get_engine
from app.ai_service import AIService, ai_search
from app.ime_service import get_ime_index
from app.telegram_bot import TelegramBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "..", "static")

_engine_ready = False
_engine_ready_time: float = 0.0


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict = {}
        self.ttl = ttl_seconds
        self.timestamps: dict = {}

    def get(self, key: str):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            del self.cache[key]
            del self.timestamps[key]
        return None

    def set(self, key: str, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear_old(self):
        now = time.time()
        expired = [k for k, t in self.timestamps.items() if now - t >= self.ttl]
        for k in expired:
            del self.cache[k]
            del self.timestamps[k]


response_cache = ResponseCache(ttl_seconds=settings.RESPONSE_CACHE_TTL)


async def _background_load():
    """Load index + warm up embedding model without blocking the event loop."""
    global _engine_ready, _engine_ready_time
    try:
        logger.info("Background: loading MedDRA engine …")
        eng = await asyncio.to_thread(get_engine)
        logger.info(f"Background: index ready ({len(eng.terms)} terms). Warming up model …")
        await asyncio.to_thread(eng.warmup)
        _engine_ready = True
        _engine_ready_time = time.time()
        logger.info("✅ Engine fully ready (index + model loaded)")
    except Exception as e:
        logger.error(f"❌ Engine background load failed: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting MedDRA Coding Assistant …")
    asyncio.create_task(_background_load())
    try:
        await TelegramBot.start()
    except Exception as e:
        logger.error(f"Telegram start failed: {e}")
    yield
    try:
        await TelegramBot.stop()
    except Exception:
        pass
    logger.info("🛑 Shutting down.")


app = FastAPI(
    title="MedDRA Coding Assistant",
    description="Hybrid search + AI accuracy layer for MedDRA coding",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K)
    ai: bool | None = Field(default=None)


class Match(BaseModel):
    term: str
    pt: str
    soc: str | None = None
    term_id: int
    level: str
    confidence: float
    semantic_score: float
    lexical_score: float
    match_type: str
    relevance_score: float = 0.0
    ai_pick: bool = False
    is_ime: bool = False


class AIInfo(BaseModel):
    used: bool = False
    interpretation: str | None = None
    reason: str | None = None
    expanded_terms: list[str] = []


class CodeResponse(BaseModel):
    query: str
    count: int
    results: list[Match]
    ai: AIInfo = AIInfo()
    processing_time_ms: float = 0.0


# ── Search ───────────────────────────────────────────────────────────────────

async def _search(text: str, top_k: int, ai: bool | None) -> CodeResponse:
    start_time = time.time()

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query")

    top_k = max(1, min(top_k, settings.MAX_TOP_K))

    cache_key = f"{text}|{top_k}|{ai}"
    cached = response_cache.get(cache_key)
    if cached:
        return cached

    # ── Guard: engine not ready yet ─────────────────────────────────────
    # We check the ready flag first. If not ready, we try get_engine() in a
    # thread (which may complete quickly if another worker already loaded it).
    # We never call get_engine() bare on the event loop — that would block
    # during the initial model download.
    if not _engine_ready:
        # Engine may still be usable for lexical-only search even if warmup
        # (model download) hasn't finished. Try it non-blocking.
        if engine_mod._engine is None:
            raise HTTPException(
                status_code=503,
                detail="Service is starting up — index loading. Retry in a few seconds.",
            )

    try:
        # engine_mod._engine is set by get_engine() during _background_load;
        # by the time we reach here it exists (guarded above).
        eng = engine_mod._engine  # never blocks — just returns the object
    except Exception as e:
        logger.error(f"Engine access failed: {e}")
        raise HTTPException(status_code=503, detail="Search engine unavailable — please retry.")

    use_ai = settings.AI_ENABLED and (settings.AI_DEFAULT if ai is None else ai)
    results = None
    ai_info = AIInfo()

    if use_ai:
        try:
            data = await ai_search(eng, text, top_k)
            results = data["results"]
            ai_info = AIInfo(**data["ai"])
        except Exception as e:
            logger.error(f"AI layer failed, falling back to hybrid: {e}")
            use_ai = False

    if not use_ai or results is None:
        results = await asyncio.to_thread(eng.search, text, top_k)

    # Enrich with IME status
    ime_idx = get_ime_index()
    for r in results:
        r["is_ime"] = ime_idx.is_ime(r["term_id"], r.get("pt"))

    processing_time = (time.time() - start_time) * 1000

    response = CodeResponse(
        query=text,
        count=len(results),
        results=results,
        ai=ai_info,
        processing_time_ms=round(processing_time, 2),
    )

    response_cache.set(cache_key, response)

    if processing_time > 1000:
        logger.warning(f"Slow search: '{text[:40]}' took {processing_time:.0f}ms")

    return response


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/api/code", response_model=CodeResponse)
async def code_post(req: CodeRequest, background_tasks: BackgroundTasks):
    if len(response_cache.cache) > 1000:
        background_tasks.add_task(response_cache.clear_old)
    return await _search(req.text, req.top_k, req.ai)


@app.get("/api/code", response_model=CodeResponse)
async def code_get(
    q: str,
    top_k: int = settings.DEFAULT_TOP_K,
    ai: bool | None = None,
    background_tasks: BackgroundTasks = None,
):
    if background_tasks and len(response_cache.cache) > 1000:
        background_tasks.add_task(response_cache.clear_old)
    return await _search(q, top_k, ai)


@app.get("/health")
def health():
    """Health check — always responds 200, reports readiness in body."""
    eng = engine_mod._engine
    uptime = time.time() - _engine_ready_time if _engine_ready else 0
    semantic_active = (
        eng.enable_semantic and eng.vectors is not None
        if eng else False
    )
    return {
        "status": "healthy",
        "ready": _engine_ready,
        "terms_loaded": len(eng.terms) if eng else 0,
        "model_loaded": bool(eng and eng._model is not None),
        "semantic_enabled": semantic_active,
        "semantic_model_failed": bool(eng and eng._model_load_failed),
        "telegram": "enabled" if settings.TELEGRAM_ENABLED else "disabled",
        "ai_layer": "enabled" if settings.AI_ENABLED else "disabled",
        "cache_size": len(response_cache.cache),
        "uptime_seconds": round(uptime, 1),
        "version": "2.1.0",
    }


@app.get("/api/stats")
def stats():
    eng = engine_mod._engine
    return {
        "engine_ready": _engine_ready,
        "terms_count": len(eng.terms) if eng else 0,
        "cache_entries": len(response_cache.cache),
        "cache_ttl_seconds": settings.RESPONSE_CACHE_TTL,
        "ai_enabled": settings.AI_ENABLED,
        "timestamp": time.time(),
    }


@app.get("/api/ai-health")
async def ai_health():
    if not settings.AI_ENABLED:
        return {"configured": False, "live": False, "detail": "AI disabled"}
    return await AIService.check_health()


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if settings.TELEGRAM_WEBHOOK_SECRET:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token", "") != settings.TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")
    try:
        await TelegramBot.handle_update(await request.json())
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    return {"ok": True}


@app.get("/")
def index():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({
        "service": "MedDRA Coding Assistant",
        "docs": "/docs",
        "api": "/api/code?q=bleeding",
        "version": "2.1.0",
    })


@app.get("/credits")
def credits_page():
    path = os.path.join(STATIC_DIR, "credits.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Credits page not found"})


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
