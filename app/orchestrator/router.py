class IntentRouter:
    """Deterministic first-pass router with explicit document-scoped RAG precedence."""

    def classify(self, message: str, metadata: dict | None = None) -> str:
        text = (message or "").lower().strip()
        metadata = metadata or {}

        # An explicit document reference is an authoritative routing signal.
        # This prevents generic questions such as ``What is SQL?`` from bypassing
        # the RAG agent when the API caller has selected a document.
        if metadata.get("document_reference"):
            return "rag"

        # Explicit external-research intent must win over incidental words such as
        # "latest" or "RAG" in the same sentence. Document references remain
        # authoritative because the caller explicitly selected a local source.
        if any(k in text for k in [
            "research", "search web", "search online", "web search",
            "investigate", "look up", "find online", "browse the web",
        ]):
            return "research"

        if any(k in text for k in [
            "document", "pdf", "file", "folder", "according to",
            "from the document", "from this document", "in the document",
            "based on the document", "from the file", "in this file",
        ]):
            return "rag"

        # "latest/current" alone is ambiguous. It is intentionally not enough
        # to invoke external research without an explicit research/search cue.

        if any(k in text for k in [
            "excel", "power bi", "powerpoint", "ppt", "word",
            "jupyter", "open ",
        ]):
            return "application"

        if any(k in text for k in ["remember", "memory", "forget"]):
            return "memory"

        if any(k in text for k in [
            "screenshot", "screen", "click", "button", "visual",
        ]):
            return "vision"

        if any(k in text for k in [
            "plan", "steps", "build", "create project", "architecture",
        ]):
            return "planning"

        return "general"
