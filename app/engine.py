"""
MedDRA Coding Engine — hybrid fuzzy + semantic search.

Fixes vs v2.0:
  ✓ ENABLE_SEMANTIC=true no longer causes 502/503:
      - Single-worker safe (fastembed lazy-loads in a thread, not blocking the event loop)
      - Graceful degradation: if model download/load fails, falls back to lexical-only
      - FASTEMBED_CACHE_DIR respected for persistent cache on Render disk
  ✓ Expanded ABBREV_MAP + new CLINICAL_PHRASE_MAP for colloquial → MedDRA language:
      "drug not working", "tablet stuck throat", "frug ineffective", "choking on pill" etc.
  ✓ Longest-match-first phrase normalization so multi-word phrases take priority
  ✓ All other logic unchanged from v2.0
"""
import gzip
import json
import logging
import os
import re
import threading
from collections import OrderedDict

import numpy as np
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MEDDRA_DATA_DIR", os.path.join(HERE, "..", "data"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Medical abbreviations ────────────────────────────────────────────────────
ABBREV_MAP: dict[str, str] = {
    # Vitals / basic
    "sob": "shortness of breath",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "o2": "oxygen saturation",
    "spo2": "oxygen saturation",
    "temp": "temperature",
    # Systems
    "gi": "gastrointestinal",
    "cns": "central nervous system",
    "cv": "cardiovascular",
    "pv": "peripheral vascular",
    "msk": "musculoskeletal",
    "resp": "respiratory",
    "gu": "genitourinary",
    "ent": "ear nose throat",
    # Conditions (abbreviated)
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "chf": "congestive heart failure",
    "mi": "myocardial infarction",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "ar": "allergic rhinitis",
    "gerd": "gastroesophageal reflux disease",
    "ibs": "irritable bowel syndrome",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "afib": "atrial fibrillation",
    "af": "atrial fibrillation",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "cad": "coronary artery disease",
    "tia": "transient ischemic attack",
    "anx": "anxiety",
    "dep": "depression",
    "ptsd": "post-traumatic stress disorder",
    "adhd": "attention deficit hyperactivity disorder",
    "ocd": "obsessive compulsive disorder",
    "bpd": "borderline personality disorder",
    # Symptoms abbreviated
    "n/v": "nausea and vomiting",
    "n/a/v": "nausea and vomiting",
    "diarr": "diarrhea",
    "vomit": "vomiting",
    "abd": "abdominal",
    "ha": "headache",
    "ha2": "headache",
    "ond": "other neurological disorder",
    "lbp": "low back pain",
    "neck pain": "neck pain",
    # Drug-related
    "ae": "adverse event",
    "adr": "adverse drug reaction",
    "sar": "serious adverse reaction",
    "susar": "suspected unexpected serious adverse reaction",
}

# ── Clinical phrase map: colloquial patient language → formal MedDRA query ──
# Longest phrase wins — sorted by length descending at load time.
# These map verbatim patient phrases to the correct MedDRA search query.
CLINICAL_PHRASE_MAP: dict[str, str] = {
    # ── Drug ineffectiveness ─────────────────────────────────────────────
    "drug not working": "drug ineffective",
    "drug is not working": "drug ineffective",
    "drug doesn't work": "drug ineffective",
    "drug does not work": "drug ineffective",
    "drug is ineffective": "drug ineffective",
    "drug stopped working": "drug ineffective",
    "drug no longer working": "drug ineffective",
    "drug lost effect": "drug ineffective",
    "drug losing effect": "drug ineffective",
    "drug not effective": "drug ineffective",
    "drug has no effect": "drug ineffective",
    "drug not helping": "drug ineffective",
    "drug doesn't help": "drug ineffective",
    "drug does not help": "drug ineffective",
    "drug isn't working": "drug ineffective",
    "drug not doing anything": "drug ineffective",
    "frug ineffective": "drug ineffective",          # common typo
    "frug not working": "drug ineffective",          # common typo
    "medication not working": "drug ineffective",
    "medication is not working": "drug ineffective",
    "medication doesn't work": "drug ineffective",
    "medication not effective": "drug ineffective",
    "medication not helping": "drug ineffective",
    "medication has no effect": "drug ineffective",
    "medication stopped working": "drug ineffective",
    "medicine not working": "drug ineffective",
    "medicine is not working": "drug ineffective",
    "medicine doesn't work": "drug ineffective",
    "medicine not effective": "drug ineffective",
    "medicine not helping": "drug ineffective",
    "medicine has no effect": "drug ineffective",
    "treatment not working": "therapeutic response decreased",
    "treatment not effective": "therapeutic response decreased",
    "treatment failed": "therapeutic response decreased",
    "no response to treatment": "therapeutic response decreased",
    "no improvement from drug": "drug ineffective",
    "no improvement from medication": "drug ineffective",
    "pill not working": "drug ineffective",
    "pill doesn't work": "drug ineffective",
    "tablet not working": "drug ineffective",
    "tablet doesn't work": "drug ineffective",
    "capsule not working": "drug ineffective",
    "injection not working": "drug ineffective",
    "patch not working": "drug ineffective",
    "drug wearing off": "drug ineffective",
    "drug wears off quickly": "drug ineffective",
    "drug tolerance": "drug tolerance",
    "drug not absorbed": "drug absorption decreased",
    # ── Dysphagia / swallowing / pill stuck ─────────────────────────────
    "tablet stuck in throat": "dysphagia",
    "tablet stuck throat": "dysphagia",
    "tablet stuck in my throat": "dysphagia",
    "tablet lodged in throat": "dysphagia",
    "tablet lodged throat": "dysphagia",
    "tablet caught in throat": "dysphagia",
    "tablet caught throat": "dysphagia",
    "tablet won't go down": "dysphagia",
    "tablet wont go down": "dysphagia",
    "tablet hard to swallow": "dysphagia",
    "tablet is hard to swallow": "dysphagia",
    "tablet too big to swallow": "dysphagia",
    "tablet difficult to swallow": "dysphagia",
    "tablet stuck in esophagus": "dysphagia",
    "tablet stuck esophagus": "dysphagia",
    "pill stuck in throat": "dysphagia",
    "pill stuck throat": "dysphagia",
    "pill stuck in my throat": "dysphagia",
    "pill lodged in throat": "dysphagia",
    "pill lodged throat": "dysphagia",
    "pill caught in throat": "dysphagia",
    "pill caught throat": "dysphagia",
    "pill won't go down": "dysphagia",
    "pill wont go down": "dysphagia",
    "pill hard to swallow": "dysphagia",
    "pill too big to swallow": "dysphagia",
    "pill difficult to swallow": "dysphagia",
    "pill stuck in esophagus": "dysphagia",
    "pill stuck esophagus": "dysphagia",
    "capsule stuck in throat": "dysphagia",
    "capsule stuck throat": "dysphagia",
    "capsule hard to swallow": "dysphagia",
    "capsule lodged throat": "dysphagia",
    "medication stuck in throat": "dysphagia",
    "medication stuck throat": "dysphagia",
    "medication caught in throat": "dysphagia",
    "medicine stuck in throat": "dysphagia",
    "medicine stuck throat": "dysphagia",
    "medicine caught in throat": "dysphagia",
    "difficulty swallowing tablet": "dysphagia",
    "difficulty swallowing pill": "dysphagia",
    "difficulty swallowing capsule": "dysphagia",
    "difficulty swallowing medication": "dysphagia",
    "trouble swallowing tablet": "dysphagia",
    "trouble swallowing pill": "dysphagia",
    "trouble swallowing capsule": "dysphagia",
    "can't swallow tablet": "dysphagia",
    "cannot swallow tablet": "dysphagia",
    "can't swallow pill": "dysphagia",
    "cannot swallow pill": "dysphagia",
    "swallowing difficulty": "dysphagia",
    "hard to swallow": "dysphagia",
    "difficult to swallow": "dysphagia",
    "choking on tablet": "choking",
    "choking on pill": "choking",
    "choking on capsule": "choking",
    "choking on medication": "choking",
    "tablet in airway": "choking",
    "pill in airway": "choking",
    "tablet stuck in airway": "choking",
    # ── Allergic / hypersensitivity ──────────────────────────────────────
    "allergic to medicine": "drug hypersensitivity",
    "allergic to medication": "drug hypersensitivity",
    "allergic to drug": "drug hypersensitivity",
    "allergic reaction to medicine": "drug hypersensitivity",
    "allergic reaction to medication": "drug hypersensitivity",
    "allergic reaction to drug": "drug hypersensitivity",
    "allergy to medicine": "drug hypersensitivity",
    "allergy to medication": "drug hypersensitivity",
    "allergy to drug": "drug hypersensitivity",
    "drug allergy": "drug hypersensitivity",
    "medication allergy": "drug hypersensitivity",
    "medicine allergy": "drug hypersensitivity",
    # ── Rash / skin ─────────────────────────────────────────────────────
    "rash from medicine": "drug eruption",
    "rash from medication": "drug eruption",
    "rash from drug": "drug eruption",
    "rash after taking medication": "drug eruption",
    "rash after taking medicine": "drug eruption",
    "skin rash from drug": "drug eruption",
    "hives from medicine": "urticaria",
    "hives from medication": "urticaria",
    "itching from medicine": "pruritus",
    "itching from medication": "pruritus",
    "itching from drug": "pruritus",
    # ── Nausea / GI from drug ─────────────────────────────────────────
    "upset stomach from drug": "nausea",
    "upset stomach from medicine": "nausea",
    "upset stomach from medication": "nausea",
    "stomach upset from drug": "nausea",
    "nausea from drug": "nausea",
    "nausea from medication": "nausea",
    "nausea from medicine": "nausea",
    "vomiting from drug": "vomiting",
    "vomiting from medication": "vomiting",
    "throwing up from medication": "vomiting",
    "diarrhea from medication": "diarrhea",
    "diarrhea from drug": "diarrhea",
    "diarrhea from medicine": "diarrhea",
    "loose stool from medication": "diarrhea",
    # ── Heart / cardiac ─────────────────────────────────────────────────
    "heart racing from medication": "tachycardia",
    "heart racing from drug": "tachycardia",
    "fast heartbeat from medication": "tachycardia",
    "fast heart rate from drug": "tachycardia",
    "irregular heartbeat from medication": "cardiac arrhythmia",
    "heart pounding from medication": "palpitations",
    "palpitations from medication": "palpitations",
    "heart fluttering from medication": "palpitations",
    # ── CNS / neurological ───────────────────────────────────────────────
    "dizzy from medication": "dizziness",
    "dizziness from medication": "dizziness",
    "dizzy from drug": "dizziness",
    "feeling dizzy": "dizziness",
    "feeling faint": "presyncope",
    "blurry vision from medication": "vision blurred",
    "blurred vision from medication": "vision blurred",
    "headache from medication": "headache",
    "headache from drug": "headache",
    "confusion from medication": "confusional state",
    "confused from medication": "confusional state",
    "memory loss from medication": "amnesia",
    "tremor from medication": "tremor",
    "shaking from medication": "tremor",
    "numbness from medication": "hypoaesthesia",
    "tingling from medication": "paraesthesia",
    # ── Sleep ────────────────────────────────────────────────────────────
    "can't sleep": "insomnia",
    "cannot sleep": "insomnia",
    "trouble sleeping": "insomnia",
    "difficulty sleeping": "insomnia",
    "unable to sleep": "insomnia",
    "waking up at night": "insomnia",
    "insomnia from medication": "insomnia",
    "sleepiness from medication": "somnolence",
    "drowsy from medication": "somnolence",
    "tired from medication": "fatigue",
    "fatigue from medication": "fatigue",
    # ── Injection site / infusion ─────────────────────────────────────
    "injection site pain": "injection site pain",
    "injection site swelling": "injection site swelling",
    "injection site redness": "injection site erythema",
    "injection site reaction": "injection site reaction",
    "infusion reaction": "infusion related reaction",
    "infusion site pain": "infusion site pain",
    # ── Common patient descriptions ──────────────────────────────────────
    "throwing up": "vomiting",
    "being sick": "vomiting",
    "feeling sick": "nausea",
    "feel sick": "nausea",
    "shortness of breath": "dyspnoea",
    "short of breath": "dyspnoea",
    "can't breathe": "dyspnoea",
    "cannot breathe": "dyspnoea",
    "difficulty breathing": "dyspnoea",
    "trouble breathing": "dyspnoea",
    "chest pain": "chest pain",
    "chest tightness": "chest discomfort",
    "chest discomfort": "chest discomfort",
    "back pain": "back pain",
    "joint pain": "arthralgia",
    "muscle pain": "myalgia",
    "muscle ache": "myalgia",
    "muscle cramps": "muscle spasms",
    "leg cramps": "muscle spasms",
    "swollen legs": "peripheral oedema",
    "swollen ankles": "peripheral oedema",
    "ankle swelling": "peripheral oedema",
    "weight gain": "weight increased",
    "weight loss": "weight decreased",
    "loss of appetite": "decreased appetite",
    "no appetite": "decreased appetite",
    "increased appetite": "increased appetite",
    "hair loss": "alopecia",
    "hair falling out": "alopecia",
    "dry mouth": "dry mouth",
    "thirst": "thirst",
    "frequent urination": "pollakiuria",
    "painful urination": "dysuria",
    "blood in urine": "haematuria",
    "blood in stool": "haematochezia",
    "dark urine": "chromaturia",
    "low blood pressure": "hypotension",
    "high blood pressure": "hypertension",
    "low blood sugar": "hypoglycaemia",
    "high blood sugar": "hyperglycaemia",
    "sweating": "hyperhidrosis",
    "night sweats": "night sweats",
    "hot flashes": "hot flush",
    "hot flush": "hot flush",
    "flushing": "flushing",
    "runny nose": "rhinorrhoea",
    "stuffy nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "sore throat": "pharyngitis",
    "mouth sores": "stomatitis",
    "mouth ulcers": "mouth ulceration",
    "canker sores": "aphthous stomatitis",
    "stomach pain": "abdominal pain",
    "abdominal pain": "abdominal pain",
    "stomach cramps": "abdominal pain",
    "bloating": "abdominal distension",
    "constipation": "constipation",
    "diarrhea": "diarrhea",
    "bloody stool": "haematochezia",
    "black stool": "melaena",
    "tarry stool": "melaena",
    "jaundice": "jaundice",
    "yellow skin": "jaundice",
    "yellow eyes": "jaundice",
    "dark circles": "periorbital oedema",
    "eye redness": "eye irritation",
    "red eyes": "eye irritation",
    "eye pain": "eye pain",
    "ear ringing": "tinnitus",
    "ringing in ears": "tinnitus",
    "hearing loss": "hearing impaired",
    "mood swings": "mood altered",
    "anxiety": "anxiety",
    "feeling anxious": "anxiety",
    "depression": "depression",
    "feeling depressed": "depressed mood",
    "suicidal thoughts": "suicidal ideation",
    "feeling suicidal": "suicidal ideation",
    "panic attacks": "panic attack",
    "panic attack": "panic attack",
    "hallucinations": "hallucination",
    "seeing things": "visual hallucination",
    "hearing things": "auditory hallucination",
}

# Pre-sort by length descending (longest phrase matches first)
_SORTED_PHRASES: list[tuple[str, str]] = sorted(
    CLINICAL_PHRASE_MAP.items(), key=lambda x: len(x[0]), reverse=True
)


class QueryCache:
    """LRU cache for search results with TTL."""

    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.timestamps: dict = {}
        self.lock = threading.Lock()

    def get(self, key: str):
        import time
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps.get(key, 0) < self.ttl:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None

    def set(self, key: str, value):
        import time
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            self.timestamps[key] = time.time()
            if len(self.cache) > self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]


class OptimizedMeddraEngine:
    """MedDRA search engine with hybrid fuzzy + semantic search."""

    def __init__(self, data_dir: str = DATA_DIR, enable_semantic: bool = True):
        self.data_dir = data_dir
        self.enable_semantic = enable_semantic
        self.terms: list[dict] = []
        self.llt_names: list[str] = []
        self.llt_lower: list[str] = []
        self.vectors: np.ndarray | None = None
        self.by_id: dict = {}
        self._model = None
        self._model_lock = threading.Lock()
        self._model_load_failed = False   # ← NEW: track permanent failure
        self.query_cache = QueryCache(max_size=1024)
        self._load()

    def _load(self):
        terms_path = os.path.join(self.data_dir, "meddra_terms.jsonl.gz")
        vec_path = os.path.join(self.data_dir, "meddra_vectors.npz")

        logger.info(f"Loading MedDRA terms from {terms_path}")
        with gzip.open(terms_path, "rt", encoding="utf-8") as f:
            self.terms = [json.loads(line) for line in f]

        self.llt_names = [t["llt"] for t in self.terms]
        self.llt_lower = [n.lower() for n in self.llt_names]
        self.by_id = {t["id"]: t for t in self.terms}
        logger.info(f"Loaded {len(self.terms)} LLT terms")

        if not self.enable_semantic:
            logger.info("LITE MODE: semantic search disabled")
            return

        if not os.path.exists(vec_path):
            logger.warning(f"Vectors file not found at {vec_path} — semantic disabled")
            self.enable_semantic = False
            return

        logger.info(f"Loading semantic vectors from {vec_path}")
        try:
            npz = np.load(vec_path, allow_pickle=True)
            q = npz["vectors"].astype(np.float32) / 127.0
            q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            self.vectors = q.astype(np.float16)
            logger.info(f"Loaded vectors {self.vectors.shape}")
        except Exception as e:
            logger.error(f"Failed to load vectors: {e} — semantic disabled")
            self.enable_semantic = False

    def _get_model(self):
        """Lazy-load the fastembed model. Thread-safe. Never raises — returns None on failure."""
        if self._model_load_failed:
            return None
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._model_load_failed:
                return None

            try:
                from fastembed import TextEmbedding
                cache_dir = os.environ.get("FASTEMBED_CACHE_DIR")
                logger.info(
                    f"Loading embedding model: {EMBED_MODEL}"
                    + (f" (cache: {cache_dir})" if cache_dir else "")
                )
                kwargs = {"cache_dir": cache_dir} if cache_dir else {}
                self._model = TextEmbedding(model_name=EMBED_MODEL, **kwargs)
                logger.info("Embedding model ready")
            except Exception as e:
                logger.error(
                    f"fastembed model load failed: {e} — "
                    "semantic search disabled for this session. "
                    "Check FASTEMBED_CACHE_DIR and disk space."
                )
                self._model_load_failed = True
                self.enable_semantic = False

        return self._model

    def _embed_query(self, text: str) -> np.ndarray | None:
        """Embed query text. Returns None if model unavailable."""
        model = self._get_model()
        if model is None:
            return None
        try:
            vec = np.array(list(model.embed([text]))[0], dtype=np.float32)
            vec /= (np.linalg.norm(vec) + 1e-9)
            return vec
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _preprocess_query(self, text: str) -> str:
        """
        Normalize clinical phrases and abbreviations.
        Priority order:
          1. CLINICAL_PHRASE_MAP  — longest match first (full phrase normalization)
          2. ABBREV_MAP           — word-level abbreviation expansion
        """
        text_lower = text.lower().strip()
        # Remove punctuation noise but keep apostrophes for contractions
        text_clean = re.sub(r"[^\w\s']", " ", text_lower).strip()

        # 1. Phrase map — longest match wins
        for phrase, replacement in _SORTED_PHRASES:
            if phrase in text_clean:
                logger.debug(f"Phrase map: '{text_clean}' → '{replacement}'")
                return replacement

        # 2. Abbreviation map — exact full-string match first
        if text_clean in ABBREV_MAP:
            return ABBREV_MAP[text_clean]

        # 3. Word-level abbreviation expansion
        words = text_clean.split()
        expanded = [ABBREV_MAP.get(w, w) for w in words]
        result = " ".join(expanded).strip()
        return result if result != text_clean else text

    def warmup(self):
        """Pre-load the embedding model (called in background thread on startup)."""
        if not self.enable_semantic:
            return
        try:
            self._embed_query("warmup query")
            logger.info("Semantic warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def search(
        self,
        text: str,
        top_k: int = 8,
        sem_candidates: int = 150,
        fuzz_candidates: int = 150,
    ) -> list[dict]:
        """Hybrid search: semantic + lexical fusion with colloquial phrase normalization."""
        text = (text or "").strip()
        if not text:
            return []

        cache_key = f"{text}|{top_k}"
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached

        processed = self._preprocess_query(text)
        q_lower = processed.lower()

        # 1. Semantic candidates
        sem_idx = np.array([], dtype=int)
        sims = None
        if self.enable_semantic and self.vectors is not None:
            qv = self._embed_query(processed)
            if qv is not None:
                try:
                    qv16 = qv.astype(np.float16)
                    sims = (self.vectors @ qv16).astype(np.float32)
                    sem_idx = np.argpartition(-sims, min(sem_candidates, len(sims) - 1))[:sem_candidates]
                except Exception as e:
                    logger.error(f"Semantic scoring failed: {e}")

        # 2. Lexical / fuzzy candidates
        fuzzy_hits = process.extract(
            q_lower,
            self.llt_lower,
            scorer=fuzz.WRatio,
            limit=fuzz_candidates,
            score_cutoff=50,
        )
        fuzz_idx = [h[2] for h in fuzzy_hits]

        # 3. Union
        cand = set(int(i) for i in sem_idx.tolist()) | set(int(i) for i in fuzz_idx)
        if not cand:
            self.query_cache.set(cache_key, [])
            return []

        # 4. Score + rank
        scored = self._score_candidates(cand, q_lower, sims)
        scored.sort(key=lambda x: x[0], reverse=True)

        # 5. Deduplicate by PT and format
        results, seen_pt = [], set()
        for rank_score, sem01, lex_conf, i, match_type in scored:
            t = self.terms[i]
            pt_key = (t.get("pt") or t["llt"]).lower()
            if pt_key in seen_pt:
                continue
            seen_pt.add(pt_key)
            results.append({
                "term": t["llt"],
                "pt": t.get("pt") or t["llt"],
                "soc": t.get("soc"),
                "term_id": t["id"],
                "level": t.get("level", "LLT"),
                "confidence": round(max(0.0, min(1.0, rank_score)) * 100, 1),
                "semantic_score": round(sem01 * 100, 1),
                "lexical_score": round(min(1.0, lex_conf) * 100, 1),
                "match_type": match_type,
                "relevance_score": round(rank_score * 100, 1),
            })
            if len(results) >= top_k:
                break

        self.query_cache.set(cache_key, results)
        return results

    def _score_candidates(self, cand, q_lower, sims):
        scored = []
        qlen = max(len(q_lower), 1)

        for i in cand:
            name_l = self.llt_lower[i]
            sem = float(sims[i]) if sims is not None else 0.0
            sem01 = max(0.0, min(1.0, sem))

            len_ratio = len(name_l) / qlen
            length_factor = max(0.4, min(1.0, 0.4 + 0.6 * len_ratio))

            token = (fuzz.token_set_ratio(q_lower, name_l) / 100.0) * length_factor
            ratio = fuzz.ratio(q_lower, name_l) / 100.0

            exact = 1.0 if q_lower == name_l else 0.0
            substr = 0.0
            if not exact and len(name_l) >= 4 and (q_lower in name_l or name_l in q_lower):
                substr = 0.85 * length_factor
            typo = ratio if ratio >= 0.8 else 0.0

            lex_conf = max(exact, typo, substr, 0.92 * token)

            if lex_conf >= 0.65 and sem01 >= 0.65:
                rank_score = max(lex_conf, sem01) * 1.0
                match_type = "semantic+lexical"
            elif lex_conf > sem01:
                rank_score = lex_conf * 0.99
                match_type = "lexical"
            else:
                rank_score = sem01 * 0.98
                match_type = "semantic"

            rank_score -= 0.0008 * len(name_l)
            scored.append((rank_score, sem01, lex_conf, i, match_type))

        return scored


# ── Singleton ────────────────────────────────────────────────────────────────

_engine: OptimizedMeddraEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> OptimizedMeddraEngine:
    """Return thread-safe singleton engine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                semantic_enabled = (
                    os.getenv("MEDDRA_SEMANTIC_ENABLED",
                              os.getenv("ENABLE_SEMANTIC", "true")).lower()
                    in ("1", "true", "yes")
                )
                _engine = OptimizedMeddraEngine(
                    data_dir=DATA_DIR,
                    enable_semantic=semantic_enabled,
                )
    return _engine
