"""
Persistent SQLite cache for RAGPropertyRetriever results.

Keyed by canonical SMILES. Stores the chemical name, the property dict
(JSON-encoded), and the paper citation list (JSON-encoded). Supports
concurrent reads/writes via sqlite3's default locking.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Optional, TYPE_CHECKING

from rdkit import Chem

if TYPE_CHECKING:
    from .rag_retrieval import RAGResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_results (
    smiles          TEXT PRIMARY KEY,
    chemical_name   TEXT,
    properties_json TEXT NOT NULL,
    citations_json  TEXT NOT NULL,
    papers_searched INTEGER NOT NULL DEFAULT 0,
    papers_with_hits INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL
);
"""


def _canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _jsonable(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class RAGCache:
    """Thin SQLite wrapper; process- and thread-safe for the usage here."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def get(self, smiles: str) -> Optional["RAGResult"]:
        canonical = _canonicalize(smiles)
        if canonical is None:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT chemical_name, properties_json, citations_json, "
                "papers_searched, papers_with_hits FROM rag_results WHERE smiles = ?",
                (canonical,),
            ).fetchone()
        if row is None:
            return None
        chemical_name, props_json, citations_json, papers_searched, papers_with_hits = row

        from .rag_retrieval import RAGResult, RetrievedProperty, PaperCitation

        props_raw = json.loads(props_json) if props_json else {}
        properties = {}
        for prop_name, payload in props_raw.items():
            if payload is None:
                properties[prop_name] = None
            else:
                properties[prop_name] = RetrievedProperty(
                    value=payload['value'],
                    source=payload.get('source', ''),
                    confidence=payload.get('confidence', 0.0),
                )

        citations = [
            PaperCitation(
                title=c.get('title', ''),
                authors=c.get('authors', []),
                doi=c.get('doi', ''),
                source_db=c.get('source_db', ''),
                properties_found=c.get('properties_found', []),
            )
            for c in json.loads(citations_json or '[]')
        ]

        return RAGResult(
            smiles=smiles,
            chemical_name=chemical_name,
            properties=properties,
            papers_searched=int(papers_searched or 0),
            papers_with_hits=int(papers_with_hits or 0),
            citations=citations,
        )

    def put(self, smiles: str, result) -> None:
        canonical = _canonicalize(smiles)
        if canonical is None:
            return
        props_payload = {
            name: (None if prop is None else _jsonable(prop))
            for name, prop in (result.properties or {}).items()
        }
        citations_payload = [_jsonable(c) for c in (result.citations or [])]
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO rag_results "
                    "(smiles, chemical_name, properties_json, citations_json, "
                    " papers_searched, papers_with_hits, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        canonical,
                        result.chemical_name,
                        json.dumps(props_payload),
                        json.dumps(citations_payload),
                        int(getattr(result, 'papers_searched', 0) or 0),
                        int(getattr(result, 'papers_with_hits', 0) or 0),
                        time.time(),
                    ),
                )
        except sqlite3.Error as e:
            logger.warning(f"RAG cache write failed: {e}")
