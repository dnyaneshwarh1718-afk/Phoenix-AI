from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class DiscoveredDocument:
    path: Path
    file_name: str
    file_type: str

class DocumentDiscovery:
    DEFAULT_SUPPORTED_EXTENSIONS = {".pdf",".txt",".md",".py",".sql",".json",".yaml",".yml",".docx",".xlsx",".xls",".pptx",".csv"}
    DEFAULT_EXCLUDED_DIRECTORIES = {"$RECYCLE.BIN","System Volume Information",".git","__pycache__",".venv","node_modules"}

    def __init__(self, supported_extensions=None, excluded_directories=None):
        extensions = supported_extensions or self.DEFAULT_SUPPORTED_EXTENSIONS
        self.supported_extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        self.excluded_directories = set(excluded_directories or self.DEFAULT_EXCLUDED_DIRECTORIES)

    def discover(self, root_path: str) -> list[DiscoveredDocument]:
        root = Path(root_path)
        if not root.exists(): raise FileNotFoundError(f"Path not found: {root_path}")
        if not root.is_dir(): raise ValueError(f"Path is not a directory: {root_path}")
        result=[]
        for current_root, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if d not in self.excluded_directories]
            for name in files:
                p=Path(current_root)/name
                ext=p.suffix.lower()
                if ext in self.supported_extensions:
                    result.append(DiscoveredDocument(p,p.name,ext))
        return sorted(result,key=lambda d:str(d.path).lower())

    def search(self, root_path: str, reference: str, limit: int = 20) -> list[DiscoveredDocument]:
        if not reference or not reference.strip() or limit <= 0: return []
        q=reference.strip().lower(); stem_q=Path(q).stem.lower(); scored=[]
        for d in self.discover(root_path):
            fn=d.file_name.lower(); stem=Path(d.file_name).stem.lower(); src=str(d.path).lower(); score=0
            if fn==q: score=100
            elif stem==stem_q: score=95
            elif fn.startswith(q): score=75
            elif stem.startswith(stem_q): score=70
            elif q in fn: score=55
            elif stem_q in fn: score=50
            elif q in src: score=20
            if score: scored.append((score,d))
        scored.sort(key=lambda x:(-x[0],str(x[1].path).lower()))
        return [d for _,d in scored[:limit]]

    def find_by_filename(self, root_path: str, file_name: str) -> list[DiscoveredDocument]:
        if not file_name or not file_name.strip(): return []
        target=file_name.strip().lower()
        return [d for d in self.discover(root_path) if d.file_name.lower()==target]

    def find_by_extension(self, root_path: str, extension: str) -> list[DiscoveredDocument]:
        ext=extension.lower(); ext=ext if ext.startswith(".") else f".{ext}"
        if ext not in self.supported_extensions: return []
        return [d for d in self.discover(root_path) if d.file_type==ext]
