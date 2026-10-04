from __future__ import annotations

import os
import re

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable
from difflib import SequenceMatcher


class MatchStatus(str, Enum):
    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


class MatchType(str, Enum):
    EXACT_FILENAME = "exact_filename"
    EXACT_STEM = "exact_stem"
    FUZZY = "fuzzy"
    NONE = "none"


@dataclass(frozen=True)
class DocumentCandidate:
    path: Path
    score: float
    match_type: MatchType

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()


@dataclass
class DocumentSearchResult:
    status: MatchStatus
    match_type: MatchType = MatchType.NONE
    path: Path | None = None
    filename: str | None = None
    extension: str | None = None
    score: float | None = None
    candidates: list[DocumentCandidate] = field(
        default_factory=list
    )
    searched_root: Path | None = None
    message: str = ""


class LocalDriveDocumentFinder:
    """
    Phoenix AI local filesystem document finder.

    Search priority:

        1. Exact filename
        2. Exact filename stem
        3. Controlled fuzzy matching
        4. Downloads fallback

    Responsibilities:

        - Locate documents on the local filesystem
        - Return deterministic search results

    This class does NOT:

        - index documents
        - generate embeddings
        - access Qdrant
        - access BM25
        - invoke an LLM
        - perform RAG
        - generate answers
    """

    # --------------------------------------------------------------
    # Supported document types
    # --------------------------------------------------------------

    DEFAULT_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".csv",
        ".md",
    }

    # --------------------------------------------------------------
    # Directories that should never be searched.
    #
    # These are normally development/build/cache directories and
    # can contain thousands of irrelevant files.
    # --------------------------------------------------------------

    SKIP_DIRECTORIES = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "coverage",
        "htmlcov",
    }

    def __init__(
        self,
        primary_root: str | Path = "R:\\",
        downloads_root: str | Path | None = None,
        supported_extensions: Iterable[str] | None = None,
        fuzzy_threshold: float = 0.82,
        ambiguity_margin: float = 0.05,
    ) -> None:

        self.primary_root = Path(primary_root)

        if downloads_root is None:
            downloads_root = Path.home() / "Downloads"

        self.downloads_root = Path(downloads_root)

        # Normalize supported extensions once.
        self.supported_extensions = {
            ext.lower()
            if ext.startswith(".")
            else f".{ext.lower()}"
            for ext in (
                supported_extensions
                if supported_extensions is not None
                else self.DEFAULT_EXTENSIONS
            )
        }

        # Normalize skipped directory names once.
        self._skip_directories = {
            directory.lower()
            for directory in self.SKIP_DIRECTORIES
        }

        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def find(self, query: str) -> DocumentSearchResult:
        """
        Search R:\\ first.

        Downloads is searched only when the primary root produces
        no meaningful result.

        Returns:

            FOUND
            AMBIGUOUS
            NOT_FOUND
        """

        normalized_query = self._normalize_query(query)

        if not normalized_query:
            return DocumentSearchResult(
                status=MatchStatus.NOT_FOUND,
                message="Document query is empty.",
            )

        # ----------------------------------------------------------
        # 1. Search primary root
        # ----------------------------------------------------------

        result = self._search_root(
            root=self.primary_root,
            query=normalized_query,
        )

        if result.status in {
            MatchStatus.FOUND,
            MatchStatus.AMBIGUOUS,
        }:
            return result

        # ----------------------------------------------------------
        # 2. Search Downloads as fallback
        # ----------------------------------------------------------

        if self.downloads_root.exists():

            result = self._search_root(
                root=self.downloads_root,
                query=normalized_query,
            )

            if result.status in {
                MatchStatus.FOUND,
                MatchStatus.AMBIGUOUS,
            }:
                return result

        # ----------------------------------------------------------
        # 3. Nothing found
        # ----------------------------------------------------------

        return DocumentSearchResult(
            status=MatchStatus.NOT_FOUND,
            message=(
                f"Could not find '{query}' on "
                f"{self.primary_root} or "
                f"{self.downloads_root}."
            ),
        )

    # ==============================================================
    # ROOT SEARCH
    # ==============================================================

    def _search_root(
        self,
        root: Path,
        query: str,
    ) -> DocumentSearchResult:

        if not root.exists():
            return DocumentSearchResult(
                status=MatchStatus.NOT_FOUND,
                searched_root=root,
                message=f"Search root does not exist: {root}",
            )

        candidates = self._discover_documents(root)

        if not candidates:
            return DocumentSearchResult(
                status=MatchStatus.NOT_FOUND,
                searched_root=root,
                message=(
                    f"No supported documents found in {root}."
                ),
            )

        # ----------------------------------------------------------
        # LEVEL 1
        # Exact filename
        # ----------------------------------------------------------

        exact_filename = [
            path
            for path in candidates
            if self._normalize_filename(path.name) == query
        ]

        if len(exact_filename) == 1:

            path = exact_filename[0]

            return self._found_result(
                path=path,
                score=1.0,
                match_type=MatchType.EXACT_FILENAME,
                root=root,
            )

        if len(exact_filename) > 1:

            return self._ambiguous_result(
                exact_filename,
                MatchType.EXACT_FILENAME,
                root,
            )

        # ----------------------------------------------------------
        # LEVEL 2
        # Exact stem
        # ----------------------------------------------------------

        query_stem = self._query_stem(query)

        exact_stem = [
            path
            for path in candidates
            if self._normalize_stem(path.stem) == query_stem
        ]

        if len(exact_stem) == 1:

            path = exact_stem[0]

            return self._found_result(
                path=path,
                score=0.98,
                match_type=MatchType.EXACT_STEM,
                root=root,
            )

        if len(exact_stem) > 1:

            return self._ambiguous_result(
                exact_stem,
                MatchType.EXACT_STEM,
                root,
            )

        # ----------------------------------------------------------
        # LEVEL 3
        # Controlled fuzzy matching
        # ----------------------------------------------------------

        fuzzy_candidates = self._fuzzy_matches(
            query_stem=query_stem,
            candidates=candidates,
        )

        if not fuzzy_candidates:

            return DocumentSearchResult(
                status=MatchStatus.NOT_FOUND,
                searched_root=root,
                message=(
                    f"No meaningful match found in {root}."
                ),
            )

        fuzzy_candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        best = fuzzy_candidates[0]

        # ----------------------------------------------------------
        # Only one candidate
        # ----------------------------------------------------------

        if len(fuzzy_candidates) == 1:

            return self._found_candidate_result(
                best,
                root,
            )

        second = fuzzy_candidates[1]

        # ----------------------------------------------------------
        # Best candidate is clearly stronger
        # ----------------------------------------------------------

        if (
            best.score >= self.fuzzy_threshold
            and (
                best.score - second.score
                >= self.ambiguity_margin
            )
        ):

            return self._found_candidate_result(
                best,
                root,
            )

        # ----------------------------------------------------------
        # Multiple strong candidates
        # ----------------------------------------------------------

        strong_candidates = [
            candidate
            for candidate in fuzzy_candidates
            if candidate.score >= self.fuzzy_threshold
        ]

        if len(strong_candidates) >= 2:

            return self._ambiguous_candidate_result(
                strong_candidates,
                root,
            )

        # ----------------------------------------------------------
        # No meaningful match
        # ----------------------------------------------------------

        return DocumentSearchResult(
            status=MatchStatus.NOT_FOUND,
            searched_root=root,
            message=(
                f"No sufficiently strong match found in {root}."
            ),
        )

    # ==============================================================
    # OPTIMIZED DOCUMENT DISCOVERY
    # ==============================================================

    def _discover_documents(
        self,
        root: Path,
    ) -> list[Path]:
        """
        Recursively discover supported documents while pruning
        irrelevant directories.

        os.walk() is used instead of Path.rglob() because os.walk()
        allows us to modify `dirs` in-place and prevent traversal
        into directories we know are irrelevant.
        """

        documents: list[Path] = []

        try:

            for current_root, dirs, files in os.walk(
                root,
                topdown=True,
                followlinks=False,
            ):

                # --------------------------------------------------
                # IMPORTANT:
                #
                # Modify dirs in-place.
                #
                # os.walk() will NOT enter directories removed here.
                # --------------------------------------------------

                dirs[:] = [
                    directory
                    for directory in dirs
                    if directory.lower()
                    not in self._skip_directories
                ]

                current_path = Path(current_root)

                for filename in files:

                    extension = Path(
                        filename
                    ).suffix.lower()

                    if extension not in self.supported_extensions:
                        continue

                    documents.append(
                        current_path / filename
                    )

        except (
            PermissionError,
            OSError,
        ):
            # Do not allow one inaccessible directory to crash
            # the entire Finder.
            pass

        return documents

    # ==============================================================
    # FUZZY MATCHING
    # ==============================================================

    def _fuzzy_matches(
        self,
        query_stem: str,
        candidates: list[Path],
    ) -> list[DocumentCandidate]:

        results: list[DocumentCandidate] = []

        query_tokens = set(query_stem.split())

        if not query_tokens:
            return results

        for path in candidates:

            candidate_stem = self._normalize_stem(
                path.stem
            )

            if not candidate_stem:
                continue

            candidate_tokens = set(
                candidate_stem.split()
            )

            common_tokens = (
                query_tokens & candidate_tokens
            )

            # No shared tokens means no meaningful match.
            if not common_tokens:
                continue

            # ------------------------------------------------------
            # Protect against weak single-token matches.
            # ------------------------------------------------------

            if (
                len(query_tokens) > 1
                and len(common_tokens) == 1
                and len(candidate_tokens) > 2
            ):
                continue

            # ------------------------------------------------------
            # Whole-name similarity
            # ------------------------------------------------------

            similarity = SequenceMatcher(
                None,
                query_stem,
                candidate_stem,
            ).ratio()

            # ------------------------------------------------------
            # Token similarity
            # ------------------------------------------------------

            token_similarity = (
                len(common_tokens)
                / max(
                    len(query_tokens),
                    len(candidate_tokens),
                )
            )

            # ------------------------------------------------------
            # Combined score
            # ------------------------------------------------------

            score = (
                0.70 * similarity
                + 0.30 * token_similarity
            )

            if score >= self.fuzzy_threshold:

                results.append(
                    DocumentCandidate(
                        path=path,
                        score=score,
                        match_type=MatchType.FUZZY,
                    )
                )

        return results

    # ==============================================================
    # NORMALIZATION
    # ==============================================================

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:

        query = query.strip()

        if not query:
            return ""

        path = Path(query)

        if path.suffix:

            return (
                LocalDriveDocumentFinder
                ._normalize_filename(path.name)
            )

        return (
            LocalDriveDocumentFinder
            ._normalize_stem(query)
        )

    @staticmethod
    def _normalize_filename(
        filename: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]+",
            "_",
            filename.lower(),
        ).strip("_")

    @staticmethod
    def _normalize_stem(
        stem: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]+",
            " ",
            stem.lower(),
        ).strip()

    @staticmethod
    def _query_stem(
        query: str,
    ) -> str:

        path = Path(query)

        if path.suffix:

            return (
                LocalDriveDocumentFinder
                ._normalize_stem(path.stem)
            )

        return (
            LocalDriveDocumentFinder
            ._normalize_stem(query)
        )

    # ==============================================================
    # RESULT HELPERS
    # ==============================================================

    @staticmethod
    def _found_result(
        path: Path,
        score: float,
        match_type: MatchType,
        root: Path,
    ) -> DocumentSearchResult:

        return DocumentSearchResult(
            status=MatchStatus.FOUND,
            match_type=match_type,
            path=path,
            filename=path.name,
            extension=path.suffix.lower(),
            score=score,
            searched_root=root,
            message=f"Document found: {path}",
        )

    @staticmethod
    def _found_candidate_result(
        candidate: DocumentCandidate,
        root: Path,
    ) -> DocumentSearchResult:

        return LocalDriveDocumentFinder._found_result(
            path=candidate.path,
            score=candidate.score,
            match_type=candidate.match_type,
            root=root,
        )

    @staticmethod
    def _ambiguous_result(
        paths: list[Path],
        match_type: MatchType,
        root: Path,
    ) -> DocumentSearchResult:

        candidates = [
            DocumentCandidate(
                path=path,
                score=1.0,
                match_type=match_type,
            )
            for path in paths
        ]

        return DocumentSearchResult(
            status=MatchStatus.AMBIGUOUS,
            match_type=match_type,
            candidates=candidates,
            searched_root=root,
            message=(
                f"Multiple documents match the request "
                f"in {root}."
            ),
        )

    @staticmethod
    def _ambiguous_candidate_result(
        candidates: list[DocumentCandidate],
        root: Path,
    ) -> DocumentSearchResult:

        return DocumentSearchResult(
            status=MatchStatus.AMBIGUOUS,
            match_type=MatchType.FUZZY,
            candidates=candidates,
            searched_root=root,
            message=(
                f"Multiple strong document matches found "
                f"in {root}."
            ),
        )