from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import textwrap
import time
import uuid
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


SUPPORTED_FILE_EXTENSIONS = {"csv", "tsv", "json", "xlsx", "xls", "parquet"}
SEARCH_TIMEOUT_SECONDS = 20
APP_DATA_DIR = Path(".datavet")
REFRESH_JOBS_FILE = APP_DATA_DIR / "refresh_jobs.json"
REFRESH_SNAPSHOTS_DIR = APP_DATA_DIR / "refresh_snapshots"
USER_FEEDBACK_MEMORY_FILE = APP_DATA_DIR / "user_feedback_memory.json"

DB_FILE = APP_DATA_DIR / "datavet.db"
USE_DB_PERSISTENCE = os.environ.get("DATAVET_USE_DB", "0") == "1"
DATABASE_URL = os.environ.get("DATAVET_DATABASE_URL", "").strip()
S3_BUCKET = os.environ.get("DATAVET_S3_BUCKET", "").strip()
S3_PREFIX = os.environ.get("DATAVET_S3_PREFIX", "datavet_snapshots/").strip() or "datavet_snapshots/"

QUALITY_SCORE_WEIGHTS = {
    "completeness": 0.35,
    "uniqueness": 0.2,
    "consistency": 0.2,
    "outlier_health": 0.25,
}

DEFAULT_RANKING_WEIGHTS = {
    "relevance": 45.0,
    "quality": 30.0,
    "freshness": 10.0,
    "access": 15.0,
    "task": 20.0,
}

TASK_KEYWORDS: dict[str, list[str]] = {
    "Auto": [],
    "Classification": ["class", "label", "category", "binary", "multiclass"],
    "Regression": ["price", "amount", "value", "forecast", "continuous"],
    "NLP": ["text", "sentence", "token", "language", "corpus", "sentiment"],
    "Computer Vision": ["image", "pixel", "bbox", "segmentation", "vision"],
    "Time Series": ["timestamp", "date", "temporal", "timeseries", "sequence"],
    "Recommendation": ["user", "item", "rating", "click", "recommend"],
    "Anomaly Detection": ["anomaly", "fraud", "outlier", "abnormal", "risk"],
}

SCHEMA_ONTOLOGY = {
    "record_id": ["id", "identifier", "uid", "recordid", "rowid"],
    "target": ["label", "class", "y", "outcome", "groundtruth"],
    "timestamp": ["date", "datetime", "time", "event_time", "created_at"],
    "amount": ["price", "cost", "value", "amount", "payment", "transaction_value"],
    "country": ["nation", "country_name", "location_country"],
}

SEMANTIC_TOPIC_MAP: dict[str, list[str]] = {
    "fraud": ["anomaly detection", "financial crime", "transaction risk", "aml"],
    "health": ["clinical", "biomedical", "diagnosis", "patient records"],
    "agriculture": ["crop yield", "soil", "weather", "food security"],
    "climate": ["temperature", "rainfall", "weather", "carbon emissions"],
    "education": ["learning outcomes", "student performance", "assessment"],
    "finance": ["credit risk", "market data", "banking", "portfolio"],
    "image": ["computer vision", "object detection", "segmentation", "classification"],
    "text": ["nlp", "sentiment", "language modeling", "classification"],
    "audio": ["speech", "acoustics", "speaker recognition", "signal processing"],
    "transport": ["mobility", "traffic", "logistics", "route optimization"],
    "energy": ["power systems", "load forecasting", "renewables", "smart grid"],
    "water": ["hydrology", "quality monitoring", "river flow", "groundwater"],
    "africa": ["zindi", "pan-african", "local development", "emerging markets"],
}


@dataclass
class DatasetCandidate:
    source: str
    title: str
    summary: str
    page_url: str
    download_url: str | None
    license_name: str | None
    size_mb: float | None
    downloads: int | None
    last_updated: str | None
    relevance: float
    quality: float
    score: float = 0.0
    format_hint: str | None = None
    semantic_relevance: float = 0.0
    task_fit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ENGINE_OPTIONS: list[dict[str, Any]] = [
    {"label": "Pandas (Python)", "implemented": True},
    {"label": "NumPy + Pandas (Python)", "implemented": True},
    {"label": "Polars (Python)", "implemented": True},
    {"label": "Dask DataFrame (Python)", "implemented": False},
    {"label": "DuckDB SQL", "implemented": True},
    {"label": "SQLite SQL", "implemented": True},
    {"label": "PySpark DataFrame", "implemented": False},
    {"label": "Spark SQL", "implemented": False},
    {"label": "Vaex", "implemented": False},
    {"label": "Modin", "implemented": False},
    {"label": "cuDF", "implemented": False},
    {"label": "R dplyr", "implemented": False},
    {"label": "R data.table", "implemented": False},
    {"label": "Julia DataFrames", "implemented": False},
    {"label": "Apache Beam", "implemented": False},
]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def token_overlap_score(topic: str, text: str) -> float:
    topic_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]+", topic.lower()) if tok}
    text_tokens = {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if tok}
    if not topic_tokens:
        return 0.0
    overlap = topic_tokens.intersection(text_tokens)
    return round(len(overlap) / len(topic_tokens), 4)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def freshness_score(updated_at: str | None) -> float:
    parsed = parse_datetime(updated_at)
    if not parsed:
        return 0.4
    now = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max((now - parsed).days, 0)
    if age_days < 180:
        return 1.0
    if age_days < 365:
        return 0.85
    if age_days < 730:
        return 0.65
    if age_days < 1460:
        return 0.5
    return 0.3


def score_candidate(candidate: DatasetCandidate) -> float:
    download_signal = 0.5
    if candidate.downloads is not None:
        download_signal = min(np.log10(max(candidate.downloads, 1) + 1) / 6.0, 1.0)

    license_signal = 1.0 if candidate.license_name else 0.5
    size_signal = 0.6
    if candidate.size_mb is not None:
        if 5 <= candidate.size_mb <= 1500:
            size_signal = 1.0
        elif candidate.size_mb < 5:
            size_signal = 0.7
        else:
            size_signal = 0.5

    freshness = freshness_score(candidate.last_updated)
    quality = (0.35 * download_signal) + (0.25 * license_signal) + (0.2 * size_signal) + (0.2 * freshness)
    candidate.quality = round(quality, 4)

    final_score = (
        45.0 * candidate.relevance
        + 30.0 * candidate.quality
        + 10.0 * freshness
        + 15.0 * (1.0 if candidate.download_url else 0.4)
    )
    return round(final_score, 2)


def safe_get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any] | None:
    try:
        resp = requests.get(url, params=params, timeout=SEARCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def safe_get_text(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> str | None:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=SEARCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def compute_task_fit(task_type: str, text: str) -> float:
    keywords = TASK_KEYWORDS.get(task_type, [])
    if not keywords:
        return 0.5
    normalized = normalize_text(text)
    hits = sum(1 for keyword in keywords if keyword in normalized)
    return round(min(hits / max(len(keywords), 1), 1.0), 4)


def ensure_app_dirs() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFRESH_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    ensure_app_dirs()
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kv_store (k TEXT PRIMARY KEY, v TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def postgres_read(key: str) -> Any | None:
    if not DATABASE_URL:
        return None
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS datavet_kv_store (k TEXT PRIMARY KEY, v JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
                )
                cur.execute("SELECT v::text FROM datavet_kv_store WHERE k = %s", (key,))
                row = cur.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
    except Exception:
        return None
    return None


def postgres_write(key: str, payload: Any) -> bool:
    if not DATABASE_URL:
        return False
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS datavet_kv_store (k TEXT PRIMARY KEY, v JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
                )
                cur.execute(
                    """
                    INSERT INTO datavet_kv_store (k, v, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (k)
                    DO UPDATE SET v = EXCLUDED.v, updated_at = NOW()
                    """,
                    (key, json.dumps(payload)),
                )
    except Exception:
        return False
    return True


def load_persisted_json(key: str, file_path: Path, default: Any) -> Any:
    pg_payload = postgres_read(key)
    if pg_payload is not None:
        return pg_payload

    if USE_DB_PERSISTENCE:
        try:
            conn = get_db_connection()
            row = conn.execute("SELECT v FROM kv_store WHERE k = ?", (key,)).fetchone()
            conn.close()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass

    ensure_app_dirs()
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_persisted_json(key: str, file_path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, indent=2)

    postgres_write(key, payload)

    if USE_DB_PERSISTENCE:
        try:
            conn = get_db_connection()
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (k, v, updated_at) VALUES (?, ?, ?)",
                (key, serialized, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    ensure_app_dirs()
    file_path.write_text(serialized, encoding="utf-8")


def upload_snapshot_to_s3_if_configured(snapshot_path: Path) -> str | None:
    if not S3_BUCKET:
        return None
    try:
        import boto3

        s3 = boto3.client("s3")
        object_key = f"{S3_PREFIX}{snapshot_path.name}"
        s3.upload_file(str(snapshot_path), S3_BUCKET, object_key)
        return f"s3://{S3_BUCKET}/{object_key}"
    except Exception:
        return None


def expand_topic_semantically(topic: str) -> list[str]:
    normalized = normalize_text(topic)
    expansions: list[str] = []

    for key, related_terms in SEMANTIC_TOPIC_MAP.items():
        if key in normalized:
            expansions.extend(related_terms)

    raw_tokens = [tok for tok in re.findall(r"[a-zA-Z0-9]+", topic.lower()) if len(tok) > 3]
    for token in raw_tokens[:3]:
        expansions.append(f"{token} dataset")
        expansions.append(f"{token} benchmark")

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for term in expansions:
        clean_term = normalize_text(term)
        if clean_term and clean_term not in seen and clean_term != normalize_text(topic):
            deduped.append(term)
            seen.add(clean_term)

    return deduped[:8]


def load_feedback_memory() -> dict[str, Any]:
    payload = load_persisted_json(
        key="feedback_memory",
        file_path=USER_FEEDBACK_MEMORY_FILE,
        default={
            "accepted_runs": 0,
            "rejected_runs": 0,
            "preferred_config": {},
            "recent_feedback": [],
        },
    )
    if isinstance(payload, dict):
        return payload
    return {
        "accepted_runs": 0,
        "rejected_runs": 0,
        "preferred_config": {},
        "recent_feedback": [],
    }


def save_feedback_memory(memory: dict[str, Any]) -> None:
    save_persisted_json("feedback_memory", USER_FEEDBACK_MEMORY_FILE, memory)


def update_feedback_memory(
    memory: dict[str, Any],
    satisfied: bool,
    config: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    if satisfied:
        memory["accepted_runs"] = int(memory.get("accepted_runs", 0)) + 1
        memory["preferred_config"] = config
    else:
        memory["rejected_runs"] = int(memory.get("rejected_runs", 0)) + 1

    recent = memory.get("recent_feedback", [])
    if not isinstance(recent, list):
        recent = []
    recent.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "satisfied": satisfied,
        "config": config,
        "note": note[:300],
    })
    memory["recent_feedback"] = recent[-15:]
    return memory


def quality_scorecard(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing = int(df.isna().sum().sum())
    completeness = max(0.0, 1.0 - (missing / total_cells))

    if rows > 0:
        uniqueness_per_col = [min(df[col].nunique(dropna=True) / rows, 1.0) for col in df.columns]
        uniqueness = float(np.mean(uniqueness_per_col)) if uniqueness_per_col else 0.0
    else:
        uniqueness = 0.0

    consistency_scores: list[float] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            consistency_scores.append(1.0)
            continue

        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            consistency_scores.append(1.0)
        else:
            lengths = series.astype(str).str.len()
            coeff = float(lengths.std() / max(lengths.mean(), 1e-6)) if len(lengths) > 1 else 0.0
            consistency_scores.append(float(max(0.0, 1.0 - min(coeff, 1.0))))

    consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

    outlier_health_scores: list[float] = []
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            outlier_health_scores.append(1.0)
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).mean()
        outlier_health_scores.append(float(max(0.0, 1.0 - outliers)))

    outlier_health = float(np.mean(outlier_health_scores)) if outlier_health_scores else 1.0

    weighted = (
        QUALITY_SCORE_WEIGHTS["completeness"] * completeness
        + QUALITY_SCORE_WEIGHTS["uniqueness"] * uniqueness
        + QUALITY_SCORE_WEIGHTS["consistency"] * consistency
        + QUALITY_SCORE_WEIGHTS["outlier_health"] * outlier_health
    )

    gates = {
        "completeness_gate": completeness >= 0.8,
        "consistency_gate": consistency >= 0.6,
        "overall_gate": weighted >= 0.7,
    }

    return {
        "completeness": round(completeness, 4),
        "uniqueness": round(uniqueness, 4),
        "consistency": round(consistency, 4),
        "outlier_health": round(outlier_health, 4),
        "overall": round(weighted, 4),
        "gates": gates,
    }


def harmonize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    harmonized = df.copy()
    report: list[str] = []

    normalized_to_original = {normalize_text(col).replace("_", ""): col for col in harmonized.columns}
    rename_map: dict[str, str] = {}

    for canonical, synonyms in SCHEMA_ONTOLOGY.items():
        for synonym in synonyms:
            key = normalize_text(synonym).replace("_", "")
            if key in normalized_to_original:
                original = normalized_to_original[key]
                if original != canonical and canonical not in harmonized.columns:
                    rename_map[original] = canonical
                    break

    if rename_map:
        harmonized = harmonized.rename(columns=rename_map)
        report.append(f"Renamed {len(rename_map)} column(s) to canonical schema names.")

    for col in harmonized.columns:
        lowered = col.lower()
        if any(token in lowered for token in ["date", "time", "timestamp"]):
            try:
                converted = pd.to_datetime(harmonized[col], errors="coerce")
                valid_ratio = float(converted.notna().mean())
                if valid_ratio >= 0.75:
                    harmonized[col] = converted
                    report.append(f"Converted '{col}' to datetime (valid ratio={valid_ratio:.2f}).")
            except Exception:
                continue

        if "amount" in lowered and harmonized[col].dtype == "object":
            cleaned = (
                harmonized[col]
                .astype(str)
                .str.replace(r"[^0-9\.-]", "", regex=True)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                harmonized[col] = parsed
                report.append(f"Normalized currency-like values in '{col}' to numeric.")

        if any(token in lowered for token in ["pct", "percent", "ratio"]) and pd.api.types.is_numeric_dtype(harmonized[col]):
            series = harmonized[col].dropna()
            if not series.empty and series.max() > 1 and series.max() <= 100:
                harmonized[col] = harmonized[col] / 100.0
                report.append(f"Scaled percentage values in '{col}' to 0-1 range.")

    if not report:
        report.append("Schema harmonization found no required adjustments.")
    return harmonized, report


def provider_uci(topic: str, limit: int) -> list[DatasetCandidate]:
    html = safe_get_text("https://archive.ics.uci.edu/datasets", params={"search": topic})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)

    results: list[DatasetCandidate] = []
    for anchor in anchors:
        href = str(anchor.get("href") or "")
        text = anchor.get_text(" ", strip=True)
        if "/dataset/" not in href or not text:
            continue
        page_url = href if href.startswith("http") else f"https://archive.ics.uci.edu{href}"
        summary = "UCI repository listing; open detail page for metadata and download files."
        candidate = DatasetCandidate(
            source="UCI Repository",
            title=text[:120],
            summary=summary,
            page_url=page_url,
            download_url=page_url,
            license_name=None,
            size_mb=None,
            downloads=None,
            last_updated=None,
            relevance=token_overlap_score(topic, text),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results


def provider_drivendata(topic: str, limit: int) -> list[DatasetCandidate]:
    html = safe_get_text("https://www.drivendata.org/search/", params={"q": topic})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)

    results: list[DatasetCandidate] = []
    for anchor in anchors:
        href = str(anchor.get("href") or "")
        title = anchor.get_text(" ", strip=True)
        if "/competitions/" not in href or not title:
            continue
        page_url = href if href.startswith("http") else f"https://www.drivendata.org{href}"
        candidate = DatasetCandidate(
            source="DrivenData",
            title=title[:120],
            summary="DrivenData challenge listing; dataset usually available inside challenge page.",
            page_url=page_url,
            download_url=page_url,
            license_name=None,
            size_mb=None,
            downloads=None,
            last_updated=None,
            relevance=token_overlap_score(topic, title),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results


def provider_zindi(topic: str, limit: int) -> list[DatasetCandidate]:
    html = safe_get_text("https://zindi.africa/competitions", params={"search": topic})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)

    results: list[DatasetCandidate] = []
    for anchor in anchors:
        href = str(anchor.get("href") or "")
        title = anchor.get_text(" ", strip=True)
        if "/competitions/" not in href or not title:
            continue
        page_url = href if href.startswith("http") else f"https://zindi.africa{href}"
        candidate = DatasetCandidate(
            source="Zindi",
            title=title[:120],
            summary="Pan-African challenge listing; open competition page for data access and rules.",
            page_url=page_url,
            download_url=page_url,
            license_name=None,
            size_mb=None,
            downloads=None,
            last_updated=None,
            relevance=token_overlap_score(topic, title),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results


def provider_tianchi(topic: str, limit: int) -> list[DatasetCandidate]:
    html = safe_get_text("https://tianchi.aliyun.com/competition/gameList/activeList", params={"search": topic})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)

    results: list[DatasetCandidate] = []
    for anchor in anchors:
        href = str(anchor.get("href") or "")
        title = anchor.get_text(" ", strip=True)
        if "/competition/" not in href or not title:
            continue
        page_url = href if href.startswith("http") else f"https://tianchi.aliyun.com{href}"
        candidate = DatasetCandidate(
            source="Alibaba Tianchi",
            title=title[:120],
            summary="Tianchi challenge listing; open competition page for download and participation details.",
            page_url=page_url,
            download_url=page_url,
            license_name=None,
            size_mb=None,
            downloads=None,
            last_updated=None,
            relevance=token_overlap_score(topic, title),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results




def parse_manual_dataset_titles(text_block: str) -> list[str]:
    lines = [line.strip() for line in text_block.splitlines()]
    return [line for line in lines if line]


def score_candidate_with_weights(
    candidate: DatasetCandidate,
    task_type: str,
    ranking_weights: dict[str, float],
) -> float:
    _ = score_candidate(candidate)
    freshness = freshness_score(candidate.last_updated)
    access = 1.0 if candidate.download_url else 0.4
    task_factor = candidate.task_fit if task_type != "Auto" else 0.5

    score = (
        ranking_weights["relevance"] * candidate.relevance
        + ranking_weights["quality"] * candidate.quality
        + ranking_weights["freshness"] * freshness
        + ranking_weights["access"] * access
        + ranking_weights["task"] * task_factor
    )
    return round(score, 2)





def pin_candidates_by_titles(candidates: list[DatasetCandidate], pinned_titles: list[str]) -> list[DatasetCandidate]:
    if not pinned_titles:
        return candidates

    title_map = {normalize_text(title): title for title in pinned_titles}
    pinned: list[DatasetCandidate] = []
    rest: list[DatasetCandidate] = []

    for candidate in candidates:
        key = normalize_text(candidate.title)
        if key in title_map and len(pinned) < 5:
            pinned.append(candidate)
        else:
            rest.append(candidate)

    return pinned + rest


def provider_huggingface(topic: str, limit: int) -> list[DatasetCandidate]:
    url = "https://huggingface.co/api/datasets"
    payload = safe_get_json(url, params={"search": topic, "limit": limit})
    if not isinstance(payload, list):
        return []

    results: list[DatasetCandidate] = []
    for item in payload:
        name = item.get("id") or "Unknown HF Dataset"
        card_data = item.get("cardData") or {}
        summary = card_data.get("summary") or item.get("description") or "No description available."
        downloads = item.get("downloads")
        license_name = card_data.get("license") or item.get("license")
        updated = item.get("lastModified")

        candidate = DatasetCandidate(
            source="Hugging Face",
            title=name,
            summary=str(summary)[:500],
            page_url=f"https://huggingface.co/datasets/{name}",
            download_url=f"https://huggingface.co/datasets/{name}",
            license_name=license_name,
            size_mb=None,
            downloads=int(downloads) if isinstance(downloads, (int, float)) else None,
            last_updated=updated,
            relevance=token_overlap_score(topic, f"{name} {summary}"),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)
    return results


def provider_openml(topic: str, limit: int) -> list[DatasetCandidate]:
    encoded_topic = quote_plus(topic)
    url = f"https://www.openml.org/api/v1/json/data/list/data_name/{encoded_topic}/limit/{limit}"
    payload = safe_get_json(url)
    if not isinstance(payload, dict):
        return []

    dataset_list = (((payload.get("data") or {}).get("dataset")) or [])
    if isinstance(dataset_list, dict):
        dataset_list = [dataset_list]

    results: list[DatasetCandidate] = []
    for item in dataset_list:
        did = str(item.get("did", "")).strip()
        name = str(item.get("name", "OpenML Dataset")).strip()
        if not did:
            continue

        num_rows = item.get("NumberOfInstances")
        num_features = item.get("NumberOfFeatures")
        summary = f"Rows: {num_rows or 'unknown'}, Features: {num_features or 'unknown'}"

        candidate = DatasetCandidate(
            source="OpenML",
            title=name,
            summary=summary,
            page_url=f"https://www.openml.org/search?type=data&id={did}",
            download_url=f"https://www.openml.org/data/get_csv/{did}/{name}.csv",
            license_name=item.get("licence"),
            size_mb=None,
            downloads=None,
            last_updated=item.get("upload_date"),
            relevance=token_overlap_score(topic, f"{name} {summary}"),
            quality=0.0,
            format_hint="csv",
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)

    return results


def configure_kaggle_credentials() -> None:
    # Prefer Streamlit secrets for cloud deployment, fallback to environment variables.
    username = None
    key = None
    try:
        if "kaggle" in st.secrets:
            username = st.secrets["kaggle"].get("username")
            key = st.secrets["kaggle"].get("key")
    except Exception:
        pass

    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def provider_kaggle(topic: str, limit: int) -> list[DatasetCandidate]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except BaseException:
        return []

    try:
        configure_kaggle_credentials()
        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=topic, page=1) or []
    except BaseException:
        return []

    results: list[DatasetCandidate] = []
    for item in datasets[:limit]:
        ref = getattr(item, "ref", None)
        title = getattr(item, "title", None) or ref or "Kaggle Dataset"
        if not ref:
            continue

        summary = getattr(item, "subtitle", "") or "Kaggle dataset listing"
        size_bytes = getattr(item, "totalBytes", None)
        size_mb = round(size_bytes / (1024 * 1024), 2) if isinstance(size_bytes, (int, float)) else None
        votes = getattr(item, "voteCount", None)
        updated = getattr(item, "lastUpdated", None)

        candidate = DatasetCandidate(
            source="Kaggle",
            title=title,
            summary=str(summary)[:400],
            page_url=f"https://www.kaggle.com/datasets/{ref}",
            download_url=f"https://www.kaggle.com/datasets/{ref}",
            license_name=None,
            size_mb=size_mb,
            downloads=int(votes) if isinstance(votes, (int, float)) else None,
            last_updated=str(updated) if updated else None,
            relevance=token_overlap_score(topic, f"{title} {summary}"),
            quality=0.0,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)

    return results


def provider_datagov(topic: str, limit: int) -> list[DatasetCandidate]:
    payload = safe_get_json(
        "https://catalog.data.gov/api/3/action/package_search",
        params={"q": topic, "rows": limit},
    )
    if not isinstance(payload, dict):
        return []

    result_block = payload.get("result") or {}
    records = result_block.get("results") or []
    results: list[DatasetCandidate] = []

    for item in records:
        title = item.get("title") or "Data.gov Dataset"
        notes = item.get("notes") or ""
        resources = item.get("resources") or []

        chosen_resource = None
        for resource in resources:
            fmt = str(resource.get("format", "")).lower()
            if fmt in SUPPORTED_FILE_EXTENSIONS and resource.get("url"):
                chosen_resource = resource
                break

        download_url = chosen_resource.get("url") if chosen_resource else item.get("url")
        fmt_hint = str(chosen_resource.get("format")).lower() if chosen_resource else None

        candidate = DatasetCandidate(
            source="Data.gov",
            title=title,
            summary=str(notes)[:500] if notes else "Public open-data listing.",
            page_url=item.get("url") or "https://catalog.data.gov/",
            download_url=download_url,
            license_name=item.get("license_title"),
            size_mb=None,
            downloads=None,
            last_updated=item.get("metadata_modified"),
            relevance=token_overlap_score(topic, f"{title} {notes}"),
            quality=0.0,
            format_hint=fmt_hint,
        )
        candidate.score = score_candidate(candidate)
        results.append(candidate)

    return results


def aggregate_datasets(
    topic: str,
    target_count: int,
    task_type: str = "Auto",
    ranking_weights: dict[str, float] | None = None,
    pinned_titles: list[str] | None = None,
) -> tuple[list[DatasetCandidate], list[str]]:
    if ranking_weights is None:
        ranking_weights = DEFAULT_RANKING_WEIGHTS.copy()
    if pinned_titles is None:
        pinned_titles = []

    expanded_terms = expand_topic_semantically(topic)
    search_terms = [topic] + expanded_terms[:2]
    per_source = max(4, target_count // 4)
    per_term = max(2, per_source // max(1, len(search_terms)))
    providers = [
        ("Hugging Face", provider_huggingface),
        ("OpenML", provider_openml),
        ("Kaggle", provider_kaggle),
        ("Data.gov", provider_datagov),
        ("UCI Repository", provider_uci),
        ("DrivenData", provider_drivendata),
        ("Zindi", provider_zindi),
        ("Alibaba Tianchi", provider_tianchi),
    ]

    all_results: list[DatasetCandidate] = []
    source_messages: list[str] = []

    source_messages.append(
        f"Semantic topic expansion: {', '.join(expanded_terms[:4]) if expanded_terms else 'none (using original topic only)'}"
    )
    source_messages.append(f"Task-aware ranking mode: {task_type}")

    for source_name, fn in providers:
        source_total = 0
        for term in search_terms:
            try:
                source_rows = fn(term, per_term)
                source_total += len(source_rows)
                all_results.extend(source_rows)
            except Exception:
                continue

        if source_total:
            source_messages.append(f"{source_name}: {source_total} dataset candidate(s) discovered")
        else:
            source_messages.append(f"{source_name}: no public results or credentials required")

    expanded_text = " ".join([topic] + expanded_terms)
    for candidate in all_results:
        content = f"{candidate.title} {candidate.summary}"
        direct_relevance = token_overlap_score(topic, content)
        semantic_relevance = token_overlap_score(expanded_text, content)
        blended_relevance = (0.7 * direct_relevance) + (0.3 * semantic_relevance)

        candidate.relevance = round(blended_relevance, 4)
        candidate.semantic_relevance = round(semantic_relevance, 4)
        candidate.task_fit = compute_task_fit(task_type, content)
        candidate.score = score_candidate_with_weights(candidate, task_type, ranking_weights)

    deduped: list[DatasetCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(all_results, key=lambda x: x.score, reverse=True):
        key = (normalize_text(candidate.source), normalize_text(candidate.title))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= target_count:
            break

    if pinned_titles:
        deduped = pin_candidates_by_titles(deduped, pinned_titles)
        source_messages.append(f"Pinned external AI baseline titles into top results: {len(pinned_titles[:5])}")

    return deduped, source_messages


def build_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return None, None

    corr = numeric_df.corr(numeric_only=True).round(3)
    melt = (
        corr.reset_index()
        .melt(id_vars="index", var_name="feature_b", value_name="correlation")
        .rename(columns={"index": "feature_a"})
    )

    import altair as alt

    heatmap = (
        alt.Chart(melt)
        .mark_rect()
        .encode(
            x=alt.X("feature_a:N", sort=None),
            y=alt.Y("feature_b:N", sort=None),
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
            tooltip=["feature_a", "feature_b", "correlation"],
        )
        .properties(height=320)
    )
    return corr, heatmap


def connector_health_check(topic: str = "machine learning") -> list[dict[str, Any]]:
    checks = [
        ("Hugging Face", provider_huggingface),
        ("OpenML", provider_openml),
        ("Kaggle", provider_kaggle),
        ("Data.gov", provider_datagov),
        ("UCI Repository", provider_uci),
        ("DrivenData", provider_drivendata),
        ("Zindi", provider_zindi),
        ("Alibaba Tianchi", provider_tianchi),
    ]

    results: list[dict[str, Any]] = []
    for name, fn in checks:
        start = time.perf_counter()
        try:
            rows = fn(topic, 1)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            results.append(
                {
                    "connector": name,
                    "status": "ok" if rows else "partial",
                    "latency_ms": elapsed_ms,
                    "sample_count": len(rows),
                    "note": "connected" if rows else "reachable but no rows",
                }
            )
        except BaseException as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            results.append(
                {
                    "connector": name,
                    "status": "fail",
                    "latency_ms": elapsed_ms,
                    "sample_count": 0,
                    "note": str(exc)[:180],
                }
            )
    return results


def build_drift_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_cols = [col for col in raw_df.columns if col in cleaned_df.columns]

    for col in common_cols:
        raw_missing = float(raw_df[col].isna().mean())
        clean_missing = float(cleaned_df[col].isna().mean())
        missing_delta = round(clean_missing - raw_missing, 6)

        drift_item: dict[str, Any] = {
            "column": col,
            "raw_missing_ratio": round(raw_missing, 6),
            "clean_missing_ratio": round(clean_missing, 6),
            "missing_delta": missing_delta,
        }

        if pd.api.types.is_numeric_dtype(raw_df[col]) and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            raw_mean = float(raw_df[col].dropna().mean()) if not raw_df[col].dropna().empty else None
            clean_mean = float(cleaned_df[col].dropna().mean()) if not cleaned_df[col].dropna().empty else None
            drift_item["raw_mean"] = round(raw_mean, 6) if raw_mean is not None else None
            drift_item["clean_mean"] = round(clean_mean, 6) if clean_mean is not None else None
            if raw_mean is None or clean_mean is None:
                drift_item["mean_shift"] = None
            else:
                drift_item["mean_shift"] = round(clean_mean - raw_mean, 6)

        rows.append(drift_item)

    return rows


def build_leakage_report(df: pd.DataFrame, target_col: str | None = None) -> list[str]:
    warnings: list[str] = []
    if target_col and target_col in df.columns:
        if df[target_col].nunique(dropna=True) <= 1:
            warnings.append("Selected target has <=1 unique value; the dataset may be unsuitable for supervised learning.")

        numeric_df = df.select_dtypes(include=["number"])
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr(numeric_only=True)[target_col].drop(labels=[target_col]).abs()
            for col, value in correlations.items():
                if value >= 0.98:
                    warnings.append(
                        f"High leakage risk: '{col}' is almost perfectly correlated with target '{target_col}' (|r|={value:.3f})."
                    )

        target_name_tokens = set(re.findall(r"[a-zA-Z0-9]+", target_col.lower()))
        for col in df.columns:
            if col == target_col:
                continue
            tokens = set(re.findall(r"[a-zA-Z0-9]+", col.lower()))
            if target_name_tokens and tokens.intersection(target_name_tokens):
                warnings.append(f"Potential label leakage by naming overlap: '{col}' resembles target '{target_col}'.")

    if not warnings:
        warnings.append("No obvious leakage pattern detected with current heuristics.")
    return warnings


def build_profile_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    corr_df, _ = build_correlation_heatmap(cleaned_df)
    drift = build_drift_report(raw_df, cleaned_df)
    leakage = build_leakage_report(cleaned_df, target_col)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_stats": dataframe_stats(raw_df),
        "cleaned_stats": dataframe_stats(cleaned_df),
        "correlation_matrix": corr_df.to_dict() if corr_df is not None else {},
        "drift_report": drift,
        "leakage_checks": leakage,
    }


def build_cleaning_script(cleaning_config: dict[str, Any]) -> str:
    engine = cleaning_config.get("engine", "Pandas (Python)")
    null_strategy = cleaning_config.get("null_strategy", "median")
    intensity = cleaning_config.get("intensity", "standard")

    return textwrap.dedent(
        f"""
        import pandas as pd

        def clean_dataset(input_path: str, output_path: str) -> None:
            df = pd.read_csv(input_path)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            df = df.drop_duplicates()

            # Null handling strategy selected in DataVet Pro.
            if "{null_strategy}" == "mean":
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = df[col].fillna(df[col].mean())
            elif "{null_strategy}" == "median":
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = df[col].fillna(df[col].median())
            elif "{null_strategy}" == "zero":
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = df[col].fillna(0)
            elif "{null_strategy}" == "drop_rows":
                df = df.dropna()
            else:
                for col in df.columns:
                    mode_series = df[col].mode()
                    if not mode_series.empty:
                        df[col] = df[col].fillna(mode_series.iloc[0])

            if "{intensity}" == "deep":
                for col in df.select_dtypes(include=["number"]).columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr and iqr == iqr:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        df[col] = df[col].clip(lower=lower, upper=upper)

            df.to_csv(output_path, index=False)


        if __name__ == "__main__":
            clean_dataset("raw_dataset.csv", "cleaned_dataset.csv")
        """
    ).strip() + f"\n\n# Generated profile: engine={engine}\n"


def build_project_bundle(
    cleaned_df: pd.DataFrame,
    report_lines: list[str],
    metadata: dict[str, Any],
    profile_report: dict[str, Any],
    cleaning_config: dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cleaned_dataset.csv", cleaned_df.to_csv(index=False))
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        archive.writestr("profile_report.json", json.dumps(profile_report, indent=2))
        archive.writestr("audit_log.txt", "\n".join(report_lines))
        archive.writestr("cleaning_script.py", build_cleaning_script(cleaning_config))

    return buffer.getvalue()


def load_refresh_jobs() -> list[dict[str, Any]]:
    content = load_persisted_json("refresh_jobs", REFRESH_JOBS_FILE, [])
    return content if isinstance(content, list) else []


def save_refresh_jobs(jobs: list[dict[str, Any]]) -> None:
    save_persisted_json("refresh_jobs", REFRESH_JOBS_FILE, jobs)


def run_refresh_job(job: dict[str, Any]) -> dict[str, Any]:
    datasets, source_messages = aggregate_datasets(
        job["topic"],
        int(job["target_count"]),
        str(job.get("task_type", "Auto")),
    )
    now = datetime.now(timezone.utc)

    snapshot_payload = {
        "job_id": job["id"],
        "topic": job["topic"],
        "ran_at": now.isoformat(),
        "source_messages": source_messages,
        "datasets": [item.to_dict() for item in datasets],
    }
    snapshot_path = REFRESH_SNAPSHOTS_DIR / f"{job['id']}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")
    s3_uri = upload_snapshot_to_s3_if_configured(snapshot_path)

    next_run = now + timedelta(hours=int(job["interval_hours"]))
    job["last_run_at"] = now.isoformat()
    job["next_run_at"] = next_run.isoformat()
    job["last_result_count"] = len(datasets)
    job["last_status"] = "ok"
    if s3_uri:
        job["last_snapshot_s3"] = s3_uri
    return job


def run_due_refresh_jobs(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    now = datetime.now(timezone.utc)
    updates: list[dict[str, Any]] = []
    logs: list[str] = []

    for job in jobs:
        next_run_raw = job.get("next_run_at")
        next_run = parse_datetime(next_run_raw) if isinstance(next_run_raw, str) else None
        if next_run is None or next_run <= now:
            try:
                updated_job = run_refresh_job(job)
                updates.append(updated_job)
                logs.append(f"Job '{job['topic']}' refreshed with {updated_job.get('last_result_count', 0)} dataset(s).")
            except Exception as exc:
                job["last_status"] = f"failed: {exc}"
                updates.append(job)
                logs.append(f"Job '{job['topic']}' failed: {exc}")
        else:
            updates.append(job)

    save_refresh_jobs(updates)
    return updates, logs


def get_portal_search_links(topic: str) -> dict[str, str]:
    encoded = quote_plus(topic)
    return {
        "DrivenData": f"https://www.drivendata.org/search/?q={encoded}",
        "Google Dataset Search": f"https://datasetsearch.research.google.com/search?query={encoded}",
        "UCI Machine Learning Repository": f"https://archive.ics.uci.edu/datasets?search={encoded}",
        "Alibaba Cloud Tianchi": f"https://tianchi.aliyun.com/competition/gameList/activeList?search={encoded}",
        "Zindi (Pan-African platform)": f"https://zindi.africa/competitions?search={encoded}",
    }


def infer_extension(url: str, format_hint: str | None = None) -> str | None:
    if format_hint and format_hint.lower() in SUPPORTED_FILE_EXTENSIONS:
        return format_hint.lower()
    path = urlparse(url).path.lower()
    for ext in SUPPORTED_FILE_EXTENSIONS:
        if path.endswith(f".{ext}"):
            return ext
    return None


def read_dataset_from_bytes(blob: bytes, extension: str) -> pd.DataFrame:
    buffer = io.BytesIO(blob)
    if extension == "csv":
        return pd.read_csv(buffer)
    if extension == "tsv":
        return pd.read_csv(buffer, sep="\t")
    if extension in {"xlsx", "xls"}:
        return pd.read_excel(buffer)
    if extension == "json":
        return pd.read_json(buffer)
    if extension == "parquet":
        return pd.read_parquet(buffer)
    raise ValueError(f"Unsupported format: {extension}")


def load_selected_dataset(candidate: DatasetCandidate) -> tuple[pd.DataFrame | None, str]:
    if not candidate.download_url:
        return None, "No direct download URL found for this dataset. Use the source page link or upload manually."

    extension = infer_extension(candidate.download_url, candidate.format_hint)
    if not extension:
        return None, "No supported direct file format found. Open the source page or upload the dataset manually."

    try:
        response = requests.get(candidate.download_url, timeout=SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        df = read_dataset_from_bytes(response.content, extension)
        return df, "Dataset loaded successfully."
    except Exception as exc:
        return None, f"Failed to load dataset automatically: {exc}"


def manual_upload_to_dataframe(uploaded_file: Any) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if filename.endswith(".json"):
        return pd.read_json(uploaded_file)
    if filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file type for manual upload.")


def apply_engine_profile(df: pd.DataFrame, engine_name: str) -> tuple[pd.DataFrame, str]:
    if engine_name == "DuckDB SQL":
        try:
            import duckdb

            deduped = duckdb.query("SELECT DISTINCT * FROM df").to_df()
            return deduped, "DuckDB SQL profile applied for deduplication."
        except Exception as exc:
            return df.copy(), f"DuckDB profile unavailable, fallback to pandas: {exc}"

    if engine_name == "SQLite SQL":
        try:
            conn = sqlite3.connect(":memory:")
            df.to_sql("dataset", conn, if_exists="replace", index=False)
            deduped = pd.read_sql_query("SELECT DISTINCT * FROM dataset", conn)
            conn.close()
            return deduped, "SQLite SQL profile applied for deduplication."
        except Exception as exc:
            return df.copy(), f"SQLite profile unavailable, fallback to pandas: {exc}"

    if engine_name == "Polars (Python)":
        try:
            import polars as pl

            polars_df = pl.from_pandas(df)
            back_to_pandas = polars_df.unique().to_pandas()
            return back_to_pandas, "Polars profile applied for deduplication."
        except Exception as exc:
            return df.copy(), f"Polars profile unavailable, fallback to pandas: {exc}"

    return df.copy(), "Pandas-compatible profile applied."


def clean_dataframe(
    df: pd.DataFrame,
    engine_name: str,
    null_strategy: str,
    intensity: str,
    user_feedback: str,
) -> tuple[pd.DataFrame, list[str]]:
    working_df = df.copy()
    report: list[str] = []

    engine_df, engine_message = apply_engine_profile(working_df, engine_name)
    working_df = engine_df
    report.append(engine_message)

    # Normalize columns to reduce naming and whitespace issues.
    new_columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", col.strip().lower()) for col in working_df.columns]
    if list(working_df.columns) != new_columns:
        working_df.columns = new_columns
        report.append("Normalized column names to snake_case style.")

    before = len(working_df)
    working_df = working_df.drop_duplicates()
    removed_duplicates = before - len(working_df)
    report.append(f"Removed {removed_duplicates} duplicate row(s).")

    object_columns = working_df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        working_df[col] = working_df[col].astype("string").str.strip()
    if len(object_columns) > 0:
        report.append(f"Trimmed leading/trailing whitespace in {len(object_columns)} text column(s).")

    for col in working_df.select_dtypes(include=["number"]).columns:
        if null_strategy == "mean":
            fill_value = working_df[col].mean()
            working_df[col] = working_df[col].fillna(fill_value)
        elif null_strategy == "median":
            fill_value = working_df[col].median()
            working_df[col] = working_df[col].fillna(fill_value)
        elif null_strategy == "zero":
            working_df[col] = working_df[col].fillna(0)
        elif null_strategy == "drop_rows":
            continue
        else:
            mode_series = working_df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else 0
            working_df[col] = working_df[col].fillna(fill_value)

    for col in working_df.select_dtypes(include=["object", "string"]).columns:
        if null_strategy == "drop_rows":
            continue
        mode_series = working_df[col].mode()
        fallback_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        working_df[col] = working_df[col].fillna(fallback_value)

    if null_strategy == "drop_rows":
        before_drop = len(working_df)
        working_df = working_df.dropna()
        report.append(f"Dropped {before_drop - len(working_df)} row(s) containing null values.")
    else:
        report.append(f"Applied null handling strategy: {null_strategy}.")

    feedback_text = user_feedback.lower()
    deep_mode = intensity == "deep" or "deep" in feedback_text

    if deep_mode:
        sparse_cols = [
            col
            for col in working_df.columns
            if working_df[col].isna().mean() > 0.6
        ]
        if sparse_cols:
            working_df = working_df.drop(columns=sparse_cols)
            report.append(f"Dropped {len(sparse_cols)} sparse column(s) with >60% null ratio.")

        numeric_cols = working_df.select_dtypes(include=["number"]).columns
        outlier_cols_processed = 0
        for col in numeric_cols:
            q1 = working_df[col].quantile(0.25)
            q3 = working_df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            working_df[col] = working_df[col].clip(lower=lower, upper=upper)
            outlier_cols_processed += 1
        if outlier_cols_processed > 0:
            report.append(f"Capped outliers using IQR clipping in {outlier_cols_processed} numeric column(s).")

        if "lower" in feedback_text or "text" in feedback_text:
            text_cols = working_df.select_dtypes(include=["string", "object"]).columns
            for col in text_cols:
                working_df[col] = working_df[col].astype("string").str.lower()
            if len(text_cols) > 0:
                report.append(f"Lowercased text values in {len(text_cols)} column(s) based on feedback.")

    report.append(f"Final dataset shape: {working_df.shape[0]} row(s), {working_df.shape[1]} column(s).")
    return working_df, report


def dataframe_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


def convert_for_download(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    if fmt == "csv":
        content = df.to_csv(index=False).encode("utf-8")
        return content, "text/csv", "cleaned_dataset.csv"
    if fmt == "tsv":
        content = df.to_csv(index=False, sep="\t").encode("utf-8")
        return content, "text/tab-separated-values", "cleaned_dataset.tsv"
    if fmt == "json":
        content = df.to_json(orient="records", indent=2).encode("utf-8")
        return content, "application/json", "cleaned_dataset.json"
    if fmt == "xlsx":
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "cleaned_dataset.xlsx"
    if fmt == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "application/octet-stream", "cleaned_dataset.parquet"
    raise ValueError("Unsupported export format.")


def init_state() -> None:
    defaults = {
        "datasets": [],
        "source_messages": [],
        "recommended_index": None,
        "selected_dataset": None,
        "raw_df": None,
        "cleaned_df": None,
        "last_report": [],
        "cleaning_config": {},
        "profile_report": {},
        "expanded_terms": [],
        "refresh_logs": [],
        "ranking_weights": DEFAULT_RANKING_WEIGHTS.copy(),
        "connector_health": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_dataset_cards(candidates: list[DatasetCandidate]) -> None:
    if not candidates:
        st.info("No dataset candidates found yet. Try a broader topic.")
        return

    for idx, candidate in enumerate(candidates, start=1):
        title = f"{idx}. {candidate.title} [{candidate.source}]"
        with st.expander(title, expanded=idx <= 3):
            st.write(candidate.summary)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Relevance", f"{candidate.relevance:.2f}")
            col2.metric("Quality", f"{candidate.quality:.2f}")
            col3.metric("Rank Score", f"{candidate.score:.2f}")
            col4.metric("Task Fit", f"{candidate.task_fit:.2f}")
            st.write(f"Semantic Match: {candidate.semantic_relevance:.2f}")

            st.write(f"License: {candidate.license_name or 'Unknown'}")
            st.write(f"Last Updated: {candidate.last_updated or 'Unknown'}")

            cta1, cta2 = st.columns(2)
            cta1.link_button("Download / Open Dataset", candidate.download_url or candidate.page_url)
            cta2.link_button("Dataset Details", candidate.page_url)


def main() -> None:
    ensure_app_dirs()
    st.set_page_config(page_title="DataVet Pro Streamlit", page_icon="data", layout="wide")
    init_state()
    feedback_memory = load_feedback_memory()
    preferred_config = feedback_memory.get("preferred_config", {}) if isinstance(feedback_memory, dict) else {}

    st.title("DataVet Pro: Dataset Discovery and Cleaning Studio")
    st.caption(
        "Search datasets by topic, compare quality signals, select one, clean with your chosen engine profile, and export in your preferred format."
    )

    with st.expander("Ranking Controls and External AI Baseline", expanded=False):
        rank_col1, rank_col2, rank_col3 = st.columns(3)
        relevance_w = rank_col1.slider(
            "Relevance Weight",
            min_value=10.0,
            max_value=70.0,
            value=float(st.session_state["ranking_weights"].get("relevance", 45.0)),
            step=1.0,
        )
        quality_w = rank_col1.slider(
            "Quality Weight",
            min_value=10.0,
            max_value=60.0,
            value=float(st.session_state["ranking_weights"].get("quality", 30.0)),
            step=1.0,
        )
        freshness_w = rank_col2.slider(
            "Freshness Weight",
            min_value=0.0,
            max_value=30.0,
            value=float(st.session_state["ranking_weights"].get("freshness", 10.0)),
            step=1.0,
        )
        access_w = rank_col2.slider(
            "Access Weight",
            min_value=0.0,
            max_value=30.0,
            value=float(st.session_state["ranking_weights"].get("access", 15.0)),
            step=1.0,
        )
        task_w = rank_col3.slider(
            "Task-Fit Weight",
            min_value=0.0,
            max_value=40.0,
            value=float(st.session_state["ranking_weights"].get("task", 20.0)),
            step=1.0,
        )

        st.session_state["ranking_weights"] = {
            "relevance": relevance_w,
            "quality": quality_w,
            "freshness": freshness_w,
            "access": access_w,
            "task": task_w,
        }

        st.write("Gemini/Kimi baseline pinning lets you force common recommendations into the first 5 results.")
        use_auto_baseline = st.checkbox("Use Gemini/Kimi proxy endpoints (if configured)", value=False)
        baseline_col1, baseline_col2 = st.columns(2)
        gemini_manual_text = baseline_col1.text_area(
            "Gemini dataset titles (one per line)",
            height=110,
            placeholder="Dataset A\nDataset B\nDataset C",
        )
        kimi_manual_text = baseline_col2.text_area(
            "Kimi dataset titles (one per line)",
            height=110,
            placeholder="Dataset A\nDataset X\nDataset C",
        )

        st.caption(
            "For strict top-5 alignment: provide or fetch both lists. The app pins common titles from Gemini and Kimi into the first five slots."
        )

    with st.container(border=True):
        topic_col, task_col, count_col, button_col = st.columns([2.0, 1.2, 0.9, 0.8])
        topic = topic_col.text_input("Topic", placeholder="Example: fraud detection, crop yield forecasting, medical imaging")
        task_options = list(TASK_KEYWORDS.keys())
        default_task = preferred_config.get("task_type", "Auto") if isinstance(preferred_config, dict) else "Auto"
        default_task_idx = task_options.index(default_task) if default_task in task_options else 0
        selected_task_type = str(task_col.selectbox("Task Type", options=task_options, index=default_task_idx))
        target_count = count_col.slider("Datasets to fetch", min_value=10, max_value=30, value=20, step=1)
        search_now = button_col.button("Find Datasets", use_container_width=True)

    predicted_expansions = expand_topic_semantically(topic) if topic.strip() else []
    if predicted_expansions:
        st.caption("Semantic expansion terms: " + ", ".join(predicted_expansions[:6]))

    if search_now:
        if not topic.strip():
            st.warning("Enter a topic before searching.")
        else:
            gemini_titles = parse_manual_dataset_titles(gemini_manual_text)
            kimi_titles = parse_manual_dataset_titles(kimi_manual_text)
            if use_auto_baseline:
                gemini_titles = gemini_titles or fetch_external_ai_recommendations("gemini", topic)
                kimi_titles = kimi_titles or fetch_external_ai_recommendations("kimi", topic)

            common_ai_titles = select_common_ai_titles(gemini_titles, kimi_titles, max_items=5)

            with st.spinner("Investigating datasets across connected sources..."):
                datasets, source_messages = aggregate_datasets(
                    topic,
                    target_count,
                    selected_task_type,
                    ranking_weights=st.session_state["ranking_weights"],
                    pinned_titles=common_ai_titles,
                )

            st.session_state.datasets = datasets
            st.session_state.source_messages = source_messages
            st.session_state.recommended_index = 0 if datasets else None
            st.session_state.selected_dataset = datasets[0].to_dict() if datasets else None
            st.session_state.expanded_terms = predicted_expansions

            if common_ai_titles:
                st.success(
                    "Top-5 baseline pinning applied using common Gemini/Kimi titles: "
                    + ", ".join(common_ai_titles[:5])
                )
            else:
                st.info(
                    "No common Gemini/Kimi baseline titles were available. Ranking is using the configured scoring weights."
                )

    if st.session_state.source_messages:
        st.subheader("Discovery Status")
        for line in st.session_state.source_messages:
            st.write(f"- {line}")

    if st.button("Run Connector Health Check", use_container_width=True):
        with st.spinner("Checking connector health and latency..."):
            st.session_state["connector_health"] = connector_health_check(topic or "machine learning")

    connector_health_rows = st.session_state.get("connector_health", [])
    if connector_health_rows:
        st.subheader("Connector Health Dashboard")
        st.dataframe(pd.DataFrame(connector_health_rows), use_container_width=True, height=220)

    if st.session_state.datasets:
        st.subheader("Top Dataset Candidates")
        ranked = sorted(st.session_state.datasets, key=lambda x: x.score, reverse=True)
        st.session_state.datasets = ranked

        top = ranked[0]
        st.success(
            f"Best guess for topic fit: {top.title} from {top.source} (score {top.score:.2f}). "
            "Ranking is based on relevance, quality, freshness, and direct access signals."
        )

        render_dataset_cards(ranked)

        choice_labels = [f"{i + 1}. {d.title} [{d.source}] score={d.score:.2f}" for i, d in enumerate(ranked)]
        selected_label = st.selectbox("Select dataset to use", options=choice_labels, index=0)
        selected_index = choice_labels.index(selected_label)
        selected = ranked[selected_index]
        st.session_state.selected_dataset = selected.to_dict()

        if st.button("Load Selected Dataset", use_container_width=True):
            with st.spinner("Loading selected dataset..."):
                df, message = load_selected_dataset(selected)
            if df is not None:
                st.session_state.raw_df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.last_report = [message]
                st.success(message)
            else:
                st.warning(message)

    st.subheader("Manual Upload Fallback")
    uploaded = st.file_uploader(
        "Upload your own file if direct source download is restricted",
        type=list(SUPPORTED_FILE_EXTENSIONS),
        accept_multiple_files=False,
    )
    if uploaded is not None:
        try:
            manual_df = manual_upload_to_dataframe(uploaded)
            st.session_state.raw_df = manual_df
            st.session_state.cleaned_df = manual_df.copy()
            st.session_state.last_report = [f"Manual upload loaded: {uploaded.name}"]
            st.success(f"Manual upload succeeded: {uploaded.name}")
        except Exception as exc:
            st.error(f"Manual upload failed: {exc}")

    if st.session_state.raw_df is not None:
        st.subheader("Cleaning Setup")
        config_col1, config_col2, config_col3 = st.columns(3)

        engine_names = [engine["label"] for engine in ENGINE_OPTIONS]
        default_engine = preferred_config.get("engine", engine_names[0]) if isinstance(preferred_config, dict) else engine_names[0]
        engine_idx = engine_names.index(default_engine) if default_engine in engine_names else 0
        engine_choice = str(config_col1.selectbox("Processing language/library", options=engine_names, index=engine_idx))

        chosen_engine = next(x for x in ENGINE_OPTIONS if x["label"] == engine_choice)
        if not chosen_engine["implemented"]:
            st.info(
                "Selected engine profile is currently emulated with pandas-compatible cleaning so your workflow continues."
            )

        null_options = ["median", "mean", "mode", "zero", "drop_rows"]
        default_null = preferred_config.get("null_strategy", "median") if isinstance(preferred_config, dict) else "median"
        null_idx = null_options.index(default_null) if default_null in null_options else 0
        null_strategy = config_col2.selectbox(
            "Null handling",
            options=null_options,
            index=null_idx,
        )
        intensity_options = ["standard", "deep"]
        default_intensity = preferred_config.get("intensity", "standard") if isinstance(preferred_config, dict) else "standard"
        intensity_idx = intensity_options.index(default_intensity) if default_intensity in intensity_options else 0
        intensity = config_col3.selectbox("Cleaning intensity", options=intensity_options, index=intensity_idx)

        feedback_text = st.text_area(
            "Optional cleaning instruction",
            placeholder="Example: handle outliers aggressively and lowercase text columns",
            height=90,
        )

        harmonize_col, process_col = st.columns(2)
        if harmonize_col.button("Apply Schema Harmonization", use_container_width=True):
            harmonized, harmonize_report = harmonize_schema(st.session_state.raw_df)
            st.session_state.raw_df = harmonized
            st.session_state.cleaned_df = harmonized.copy()
            st.session_state.last_report = harmonize_report
            st.success("Schema harmonization applied.")

        if process_col.button("Process and Clean Dataset", type="primary", use_container_width=True):
            with st.spinner("Running selected cleaning pipeline..."):
                cleaned, report = clean_dataframe(
                    st.session_state.raw_df,
                    engine_choice,
                    null_strategy,
                    intensity,
                    feedback_text,
                )
            st.session_state.cleaned_df = cleaned
            st.session_state.last_report = report
            st.session_state.cleaning_config = {
                "engine": engine_choice,
                "null_strategy": null_strategy,
                "intensity": intensity,
                "feedback": feedback_text,
                "task_type": selected_task_type,
            }
            st.success("Cleaning completed.")

    if st.session_state.cleaned_df is not None:
        st.subheader("Cleaned Dataset Preview")

        raw_stats = dataframe_stats(st.session_state.raw_df)
        clean_stats = dataframe_stats(st.session_state.cleaned_df)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Rows", f"{clean_stats['rows']}", delta=clean_stats["rows"] - raw_stats["rows"])
        s2.metric("Columns", f"{clean_stats['columns']}", delta=clean_stats["columns"] - raw_stats["columns"])
        s3.metric("Duplicates", f"{clean_stats['duplicates']}", delta=clean_stats["duplicates"] - raw_stats["duplicates"])
        s4.metric("Missing Values", f"{clean_stats['missing_values']}", delta=clean_stats["missing_values"] - raw_stats["missing_values"])
        s5.metric("Memory MB", f"{clean_stats['memory_mb']}", delta=round(clean_stats["memory_mb"] - raw_stats["memory_mb"], 2))

        st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True, height=340)

        st.subheader("Cleaning Report")
        for line in st.session_state.last_report:
            st.write(f"- {line}")

        st.subheader("Data Quality Scorecard")
        scorecard = quality_scorecard(st.session_state.cleaned_df)
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Completeness", f"{scorecard['completeness']:.2f}")
        q2.metric("Uniqueness", f"{scorecard['uniqueness']:.2f}")
        q3.metric("Consistency", f"{scorecard['consistency']:.2f}")
        q4.metric("Outlier Health", f"{scorecard['outlier_health']:.2f}")
        q5.metric("Overall", f"{scorecard['overall']:.2f}")

        gate_messages = [
            f"Completeness Gate: {'PASS' if scorecard['gates']['completeness_gate'] else 'FAIL'}",
            f"Consistency Gate: {'PASS' if scorecard['gates']['consistency_gate'] else 'FAIL'}",
            f"Overall Gate: {'PASS' if scorecard['gates']['overall_gate'] else 'FAIL'}",
        ]
        for msg in gate_messages:
            st.write(f"- {msg}")

        st.subheader("Download Cleaned Dataset")
        export_format = st.selectbox("Export format", options=["csv", "xlsx", "json", "parquet", "tsv"], index=0)
        blob, mime, file_name = convert_for_download(st.session_state.cleaned_df, export_format)
        st.download_button(
            label=f"Download as {export_format.upper()}",
            data=blob,
            file_name=file_name,
            mime=mime,
            use_container_width=True,
        )

        st.subheader("Quality Check")
        opinion = st.radio("Are you satisfied with this cleaned dataset?", options=["Yes", "No"], horizontal=True)
        if opinion == "No":
            refinement_note = st.text_area(
                "Tell the model what to improve",
                placeholder="Example: reduce outliers further, keep only complete records, normalize text",
                height=100,
                key="refinement_note",
            )
            if st.button("Run Deep Re-Clean", use_container_width=True):
                with st.spinner("Applying deep re-clean based on your feedback..."):
                    refined, refined_report = clean_dataframe(
                        st.session_state.cleaned_df,
                        "Pandas (Python)",
                        "median",
                        "deep",
                        refinement_note,
                    )
                st.session_state.cleaned_df = refined
                st.session_state.last_report = refined_report
                st.session_state.cleaning_config = {
                    "engine": "Pandas (Python)",
                    "null_strategy": "median",
                    "intensity": "deep",
                    "feedback": refinement_note,
                    "task_type": selected_task_type,
                }
                updated_memory = update_feedback_memory(
                    feedback_memory,
                    satisfied=False,
                    config=st.session_state.cleaning_config,
                    note=refinement_note,
                )
                save_feedback_memory(updated_memory)
                st.success("Deep re-clean completed. Review the updated preview and report.")
        else:
            if st.button("Save This Configuration as Preferred", use_container_width=True):
                memory_update = update_feedback_memory(
                    feedback_memory,
                    satisfied=True,
                    config=st.session_state.get("cleaning_config", {}),
                    note="User accepted current cleaned dataset.",
                )
                save_feedback_memory(memory_update)
                st.success("Preference memory updated. Future runs will preload this configuration.")

        st.subheader("Advanced Profiling Reports")
        profile_target_col = st.selectbox(
            "Target column for leakage checks (optional)",
            options=["<none>"] + list(st.session_state.cleaned_df.columns),
            index=0,
        )
        target_col = None if profile_target_col == "<none>" else profile_target_col

        corr_df, corr_chart = build_correlation_heatmap(st.session_state.cleaned_df)
        if corr_chart is not None:
            st.write("Correlation Heatmap")
            st.altair_chart(corr_chart, use_container_width=True)
        else:
            st.info("Correlation heatmap requires at least two numeric columns.")

        drift_rows = build_drift_report(st.session_state.raw_df, st.session_state.cleaned_df)
        st.write("Drift Report (raw vs cleaned)")
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, height=220)

        leakage_warnings = build_leakage_report(st.session_state.cleaned_df, target_col)
        st.write("Leakage Checks")
        for warning in leakage_warnings:
            st.write(f"- {warning}")

        st.session_state.profile_report = build_profile_report(
            st.session_state.raw_df,
            st.session_state.cleaned_df,
            target_col,
        )

        st.subheader("Project Export (Reproducibility Bundle)")
        selected_dataset_state = st.session_state.get("selected_dataset")
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selected_dataset": selected_dataset_state if isinstance(selected_dataset_state, dict) else {},
            "topic_expansions": st.session_state.get("expanded_terms", []),
            "raw_stats": raw_stats,
            "cleaned_stats": clean_stats,
            "cleaning_config": st.session_state.get("cleaning_config", {}),
        }
        bundle_blob = build_project_bundle(
            st.session_state.cleaned_df,
            st.session_state.last_report,
            metadata,
            st.session_state.profile_report,
            st.session_state.get("cleaning_config", {}),
        )
        st.download_button(
            label="Download Project Bundle (.zip)",
            data=bundle_blob,
            file_name="datavet_project_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.subheader("Additional Sources")
    selected_dataset_state = st.session_state.get("selected_dataset")
    if isinstance(selected_dataset_state, dict) and selected_dataset_state.get("title"):
        lookup_topic = str(selected_dataset_state["title"])
    else:
        lookup_topic = "machine learning"

    st.write("Use these portals for broader discovery where public APIs are limited or unavailable.")
    portal_links = get_portal_search_links(lookup_topic)
    portal_cols = st.columns(2)
    for i, (name, link) in enumerate(portal_links.items()):
        portal_cols[i % 2].link_button(name, link, use_container_width=True)

    st.subheader("Scheduled Refresh Jobs")
    saved_jobs = load_refresh_jobs()

    job_col1, job_col2, job_col3, job_col4 = st.columns([1.8, 0.8, 0.8, 1.3])
    default_topic = lookup_topic if isinstance(lookup_topic, str) else "machine learning"
    refresh_topic = job_col1.text_input("Refresh topic", value=default_topic, key="refresh_topic")
    refresh_count = job_col2.number_input("Result count", min_value=10, max_value=30, value=20, step=1)
    refresh_interval = job_col3.number_input("Interval (hours)", min_value=1, max_value=168, value=24, step=1)
    refresh_task_type = job_col4.selectbox("Refresh task mode", options=list(TASK_KEYWORDS.keys()), index=0)

    add_job_col, run_due_col = st.columns(2)
    if add_job_col.button("Add Refresh Job", use_container_width=True):
        now = datetime.now(timezone.utc)
        job = {
            "id": uuid.uuid4().hex[:10],
            "topic": refresh_topic,
            "target_count": int(refresh_count),
            "interval_hours": int(refresh_interval),
            "task_type": refresh_task_type,
            "next_run_at": now.isoformat(),
            "last_run_at": None,
            "last_result_count": 0,
            "last_status": "scheduled",
        }
        saved_jobs.append(job)
        save_refresh_jobs(saved_jobs)
        st.success("Refresh job added.")
        st.rerun()

    if run_due_col.button("Run Due Jobs Now", use_container_width=True):
        updated_jobs, logs = run_due_refresh_jobs(saved_jobs)
        st.session_state.refresh_logs = logs
        save_refresh_jobs(updated_jobs)
        st.success("Due refresh jobs executed.")
        st.rerun()

    saved_jobs = load_refresh_jobs()
    if saved_jobs:
        jobs_df = pd.DataFrame(saved_jobs)
        st.dataframe(jobs_df, use_container_width=True, height=220)

        delete_options = [f"{job['id']} | {job['topic']}" for job in saved_jobs]
        selected_delete = st.selectbox("Delete a refresh job", options=delete_options)
        if st.button("Delete Selected Job", use_container_width=True):
            selected_id = selected_delete.split(" | ")[0]
            saved_jobs = [job for job in saved_jobs if job["id"] != selected_id]
            save_refresh_jobs(saved_jobs)
            st.success("Refresh job deleted.")
            st.rerun()
    else:
        st.info("No scheduled jobs yet.")

    if st.session_state.refresh_logs:
        st.write("Recent scheduler activity")
        for line in st.session_state.refresh_logs:
            st.write(f"- {line}")

    st.subheader("User Feedback Memory")
    current_memory = load_feedback_memory()
    mem_col1, mem_col2, mem_col3 = st.columns(3)
    mem_col1.metric("Accepted Runs", int(current_memory.get("accepted_runs", 0)))
    mem_col2.metric("Rejected Runs", int(current_memory.get("rejected_runs", 0)))
    mem_col3.metric("Stored Notes", len(current_memory.get("recent_feedback", [])))
    preferred = current_memory.get("preferred_config", {})
    if preferred:
        st.write("Preferred configuration loaded from feedback memory:")
        st.json(preferred)

    if st.button("Clear Feedback Memory", use_container_width=True):
        save_feedback_memory({
            "accepted_runs": 0,
            "rejected_runs": 0,
            "preferred_config": {},
            "recent_feedback": [],
        })
        st.success("Feedback memory cleared.")
        st.rerun()

    st.subheader("Delivery Status")
    st.write("- Semantic topic expansion: implemented.")
    st.write("- Profiling reports (correlation, drift, leakage): implemented.")
    st.write("- Reproducibility project export bundle: implemented.")
    st.write("- Scheduled refresh job system with snapshots: implemented.")


if __name__ == "__main__":
    main()
