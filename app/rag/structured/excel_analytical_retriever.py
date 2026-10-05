from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructuredRetrievalResult:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
    retrieval_method: str = "structured_excel"
    metadata: dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 1.0
    lexical_score: float = 1.0


class ExcelAnalyticalRetriever:
    """Deterministic structured retrieval for analytical Excel questions.

    Vector/BM25 retrieval is optimized for finding relevant text. It is not a
    reliable execution engine for questions that require aggregation or
    comparison over spreadsheet rows. This component executes a narrow,
    read-only set of analytical operations directly against the requested
    workbook and returns the computed row(s) as grounded evidence.
    """

    _OP_PATTERNS = {
        "max": re.compile(r"\b(highest|maximum|max|largest|top)\b", re.I),
        "min": re.compile(r"\b(lowest|minimum|min|smallest|bottom)\b", re.I),
        "sum": re.compile(r"\b(total|sum|combined)\b", re.I),
        "mean": re.compile(r"\b(average|mean|avg)\b", re.I),
    }
    _TOP_N = re.compile(r"\btop\s+(\d+)\b", re.I)
    _TOKEN_RE = re.compile(r"\b[a-z0-9_]+\b", re.I)

    def search(self, query: str, source_path: str) -> list[StructuredRetrievalResult]:
        if not query or not source_path:
            return []
        path = Path(source_path)
        if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"} or not path.exists():
            return []

        operation = self._detect_operation(query)
        if operation is None:
            return []

        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception:
            return []

        target_hint = self._target_hint(query, operation)
        results: list[RerankedResult] = []
        for sheet_name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            frame = frame.copy().fillna("")
            numeric = self._numeric_columns(frame)
            target = self._choose_target_column(numeric, target_hint)
            if target is None:
                continue

            values = pd.to_numeric(frame[target], errors="coerce")
            valid = frame.loc[values.notna()].copy()
            valid["__phoenix_numeric"] = values.loc[values.notna()]
            if valid.empty:
                continue

            n = self._TOP_N.search(query)
            top_n = int(n.group(1)) if n else 1
            if operation == "max":
                selected = valid.sort_values("__phoenix_numeric", ascending=False).head(top_n)
            elif operation == "min":
                selected = valid.sort_values("__phoenix_numeric", ascending=True).head(top_n)
            elif operation == "sum":
                total = float(valid["__phoenix_numeric"].sum())
                text = self._format_summary(path, sheet_name, target, "total", total, len(valid))
                results.append(self._result(path, sheet_name, text, "sum", target))
                continue
            else:
                mean = float(valid["__phoenix_numeric"].mean())
                text = self._format_summary(path, sheet_name, target, "average", mean, len(valid))
                results.append(self._result(path, sheet_name, text, "mean", target))
                continue

            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                entity_col = self._choose_entity_column(frame, target)
                entity = str(row.get(entity_col, "")).strip() if entity_col else ""
                value = row["__phoenix_numeric"]
                text = (
                    f"Structured Excel analysis. Sheet: {sheet_name}. "
                    f"Operation: {operation}. Target column: {target}. "
                    f"Rank: {rank}. {entity_col or 'Row'}: {entity}. "
                    f"{target}: {self._format_number(value)}. "
                    f"Source workbook: {path.name}."
                )
                results.append(self._result(path, sheet_name, text, operation, target, rank))

            if results:
                return results
        return []

    @classmethod
    def _detect_operation(cls, query: str) -> str | None:
        for operation, pattern in cls._OP_PATTERNS.items():
            if pattern.search(query):
                return operation
        return None

    @classmethod
    def _target_hint(cls, query: str, operation: str) -> str | None:
        tokens = cls._TOKEN_RE.findall(query.lower())
        stop = {
            "what", "which", "product", "products", "has", "have", "the", "highest",
            "lowest", "maximum", "minimum", "largest", "smallest", "top", "bottom",
            "revenue", "is", "are", "of", "by", "for", "in", "excel", "spreadsheet",
            "according", "to", "sheet", "total", "sum", "average", "mean", "from",
        }
        # Prefer common analytical field names explicitly present in the query.
        for candidate in ("revenue", "sales", "amount", "price", "cost", "quantity", "profit"):
            if candidate in tokens:
                return candidate
        candidates = [t for t in tokens if t not in stop and len(t) > 2]
        return candidates[-1] if candidates else None

    @staticmethod
    def _numeric_columns(frame: pd.DataFrame) -> list[str]:
        columns = []
        for column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().sum() > 0:
                columns.append(str(column))
        return columns

    @staticmethod
    def _choose_target_column(columns: list[str], hint: str | None) -> str | None:
        if not columns:
            return None
        if hint:
            for column in columns:
                if hint.lower() == str(column).strip().lower() or hint.lower() in str(column).lower():
                    return str(column)
        return str(columns[0])

    @staticmethod
    def _choose_entity_column(frame: pd.DataFrame, target: str) -> str | None:
        preferred = ("product", "name", "item", "model", "id")
        for pref in preferred:
            for column in frame.columns:
                if pref in str(column).strip().lower() and str(column) != target:
                    return str(column)
        for column in frame.columns:
            if str(column) != target:
                converted = pd.to_numeric(frame[column], errors="coerce")
                if converted.notna().sum() < len(frame) * 0.5:
                    return str(column)
        return None

    @staticmethod
    def _format_number(value: Any) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.6g}"

    @classmethod
    def _format_summary(cls, path: Path, sheet: str, target: str, operation: str, value: float, count: int) -> str:
        return (
            f"Structured Excel analysis. Sheet: {sheet}. Operation: {operation}. "
            f"Target column: {target}. Result: {target} {operation} = {cls._format_number(value)} "
            f"across {count} numeric rows. Source workbook: {path.name}."
        )

    @staticmethod
    def _result(path: Path, sheet: str, text: str, operation: str, target: str, rank: int = 1) -> StructuredRetrievalResult:
        digest = hashlib.sha256(f"{path.resolve()}|{sheet}|{operation}|{target}|{rank}".encode()).hexdigest()[:16]
        return StructuredRetrievalResult(
            chunk_id=f"structured-excel-{digest}",
            document_id=hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16],
            chunk_index=-1,
            text=text,
            score=1.0,
            retrieval_method="structured_excel",
            metadata={
                "source_path": str(path.resolve()),
                "file_name": path.name,
                "sheet": sheet,
                "structured_operation": operation,
                "structured_target": target,
            },
            rrf_score=1.0,
            lexical_score=1.0,
        )
