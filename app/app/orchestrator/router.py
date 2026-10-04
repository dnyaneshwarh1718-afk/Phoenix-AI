import re


class IntentRouter:
    """Deterministic first-pass router with explicit source and action precedence."""

    def classify(self, message: str, metadata: dict | None = None) -> str:
        text = (message or "").lower().strip()
        metadata = metadata or {}

        # 1. Explicitly selected documents are authoritative for knowledge retrieval.
        if metadata.get("document_reference"):
            return "rag"

        # 2. Explicit web/research intent.
        if any(k in text for k in (
            "research", "search web", "search online", "web search",
            "investigate", "look up", "find online", "browse the web",
        )):
            return "research"

        # 3. Desktop/application control must win over generic "file/document" words.
        #    "Open this Excel file" is an application operation, not a RAG query.
        if self._is_application_request(text, metadata):
            return "application"

        # 4. Knowledge retrieval / document questions.
        #
        # Project-knowledge questions are RAG questions even when the user
        # does not explicitly say "document". This is important for queries
        # such as "What vector database does Phoenix AI use?". Routing those
        # to the general LLM executor would make the answer dependent on model
        # prior knowledge and can also trigger an unnecessary Ollama call.
        if any(k in text for k in (
            "document", "pdf", "file", "folder", "according to",
            "from the document", "from this document", "in the document",
            "based on the document", "from the file", "in this file",
        )):
            return "rag"

        # 5. Explicit memory operations.
        # Memory intent must be evaluated before Phoenix/project knowledge
        # keywords can force a request into RAG. For example,
        # "Remember that my Phoenix AI vector database is Qdrant" contains
        # both a memory verb and "Phoenix AI"; the memory verb is authoritative.
        if any(k in text for k in ("remember", "memory", "forget")):
            return "memory"

        # Natural-language recall statements often omit the literal word
        # "memory". Treat first-person recall markers as memory intent while
        # leaving ordinary factual questions such as "What vector database
        # does Phoenix AI use?" routed to RAG.
        if any(k in text for k in (
            "what did i say", "what i said", "did i say",
            "my primary choice", "my choice", "i previously said",
            "i told you", "i mentioned", "you remember",
        )):
            return "memory"


        # Known Phoenix/project knowledge terms are authoritative RAG signals.
        # Keep this list intentionally narrow so ordinary general questions
        # are not hijacked by the RAG agent.
        project_knowledge_terms = (
            "phoenix ai", "phoenix-ai", "qdrant", "vector database",
            "vector db", "qwen3", "qwen 3", "ollama", "nomic-embed",
            "bm25", "reciprocal rank fusion", "rag agent",
            "memory agent", "planning agent", "application control agent",
            "computer vision agent", "research agent", "orchestrator agent",
            "phoenix architecture", "phoenix rag",
        )
        if any(term in text for term in project_knowledge_terms):
            return "rag"

        # 6. Vision / GUI perception.
        if any(k in text for k in ("screenshot", "screen", "click", "button", "visual")):
            return "vision"

        # 7. Planning.
        if any(k in text for k in ("plan", "steps", "build", "create project", "architecture")):
            return "planning"

        return "general"

    @staticmethod
    def _is_application_request(text: str, metadata: dict) -> bool:
        apps = (
            "excel", "microsoft excel", "word", "microsoft word",
            "powerpoint", "power point", "ppt", "power bi", "powerbi",
            "jupyter", "jupyter notebook",
        )
        has_app = any(a in text for a in apps) or isinstance(metadata.get("application"), str)
        if not has_app:
            return False

        # A question explicitly asking for knowledge *from* a workbook/document
        # remains RAG unless an action verb makes it a desktop operation.
        action_verbs = (
            "open", "launch", "start", "run", "close", "quit", "exit",
            "create", "edit", "modify", "update", "save", "export",
            "import", "analyze", "analyse", "inspect", "profile",
        )
        return any(re.search(rf"\b{re.escape(v)}\b", text) for v in action_verbs)
