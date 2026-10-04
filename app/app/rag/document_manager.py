
from pathlib import Path

from app.rag.ingestion.loader_factory import LoaderFactory
from app.rag.models import Document


class DocumentManager:
    def __init__(self):
        self.loader_factory = LoaderFactory()

    def load_document(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        loader = self.loader_factory.get_loader(str(path))
        return loader.load(str(path))

    def scan_folder(self, folder_path: str) -> list[Path]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder.is_dir():
            raise ValueError(f"Not a folder: {folder_path}")

        supported = set()
        for loader in self.loader_factory.loaders:
            supported.update(loader.supported_extensions)

        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )
