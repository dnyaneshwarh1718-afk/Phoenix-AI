from abc import ABC, abstractmethod

from app.rag.models import Document


class BaseDocumentLoader(ABC):

    supported_extensions: set[str] = set()

    @abstractmethod
    def load(
        self,
        file_path: str,
    ) -> Document:

        raise NotImplementedError

    def supports(
        self,
        extension: str,
    ) -> bool:

        return (
            extension.lower()
            in self.supported_extensions
        )