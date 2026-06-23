"""Data loading and graph construction for DeepModule.

The implementation mirrors the revised manuscript at an executable level:
- Java source files are converted into nodes.
- Directed dependencies are extracted from imports, type references, inheritance,
  object creation, and simple method-call qualifiers.
- Node features are semantic vectors. CodeBERT is used when available; otherwise
  a deterministic hashed semantic encoder is used so the pipeline remains
  reproducible in offline environments and CI.
- Generated/test/vendor/build artifacts and files shorter than the configured
  non-comment LOC threshold are excluded and logged.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # optional dependency used in full experiments
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - expected in lightweight CI
    AutoModel = None
    AutoTokenizer = None

try:  # optional parser used in full experiments
    import javalang  # type: ignore
except Exception:  # pragma: no cover - expected in lightweight CI
    javalang = None


@dataclass
class SimpleData:
    """Small torch-geometric-like data container.

    The original code depended directly on torch_geometric.data.Data. The sandbox
    and many CI environments do not include PyG, so this container implements the
    subset needed by the trainer while remaining compatible with attributes used
    in the paper pipeline.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    class_names: List[str]
    file_paths: List[str]
    dependency_types: List[str]
    preprocessing_log: List[Dict[str, str]]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device: torch.device | str) -> "SimpleData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


class JavaProjectDataset:
    """Build an attributed directed dependency graph from a Java project."""

    DEFAULT_EXCLUDE_DIRS = {
        ".git", ".gradle", ".mvn", "target", "build", "out", "bin",
        "node_modules", "vendor", "generated", "test", "tests", "__MACOSX",
    }

    def __init__(
        self,
        root_dir: str,
        min_non_comment_loc: int = 10,
        use_codebert: bool = True,
        embedding_dim: int = 768,
        device: Optional[str] = None,
        language: str = "java",
    ) -> None:
        self.root_dir = root_dir
        self.min_non_comment_loc = int(min_non_comment_loc)
        self.embedding_dim = int(embedding_dim)
        self.language = language
        self.extensions = {".java"} if language == "java" else {".java", ".py", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
        self.file_paths: List[str] = []
        self.class_names: List[str] = []
        self.node_mapping: Dict[str, int] = {}
        self.preprocessing_log: List[Dict[str, str]] = []
        self.dependency_types: List[str] = []
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = None
        self.bert_model = None
        self.embedding_backend = "hash"
        if use_codebert and AutoTokenizer is not None and AutoModel is not None:
            try:
                print("Loading CodeBERT model for semantic embeddings...")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
                self.bert_model.to(self.device)
                self.bert_model.eval()
                self.embedding_backend = "codebert"
            except Exception as exc:
                print(f"Warning: CodeBERT unavailable ({exc}). Falling back to deterministic hashed embeddings.")
        else:
            print("Using deterministic hashed semantic embeddings (CodeBERT unavailable or disabled).")

    @staticmethod
    def _non_comment_loc(text: str) -> int:
        in_block = False
        count = 0
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if in_block:
                if "*/" in line:
                    in_block = False
                    line = line.split("*/", 1)[1].strip()
                else:
                    continue
            if line.startswith("/*"):
                in_block = "*/" not in line
                continue
            if line.startswith("//") or line.startswith("*"):
                continue
            count += 1
        return count

    @staticmethod
    def _entity_name_from_source(path: str, text: str) -> str:
        match = re.search(r"\b(?:class|interface|enum|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        if match:
            return match.group(1)
        return os.path.splitext(os.path.basename(path))[0]

    def _is_excluded_path(self, path: str) -> bool:
        parts = set(os.path.normpath(path).split(os.sep))
        return bool(parts & self.DEFAULT_EXCLUDE_DIRS)

    def parse_files(self) -> None:
        """Scan the project and collect architecture-relevant Java entities."""
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"Path is not a directory: {self.root_dir}")

        self.file_paths.clear()
        self.class_names.clear()
        self.node_mapping.clear()
        self.preprocessing_log.clear()

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.DEFAULT_EXCLUDE_DIRS]
            if self._is_excluded_path(root):
                continue
            for file_name in sorted(files):
                if os.path.splitext(file_name)[1] not in self.extensions:
                    continue
                path = os.path.join(root, file_name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError as exc:
                    self.preprocessing_log.append({"file": path, "action": "excluded", "reason": f"read_error:{exc}"})
                    continue

                loc = self._non_comment_loc(text)
                if loc < self.min_non_comment_loc:
                    self.preprocessing_log.append({"file": path, "action": "excluded", "reason": f"short_file:{loc}"})
                    continue
                if not text.strip():
                    self.preprocessing_log.append({"file": path, "action": "excluded", "reason": "empty_file"})
                    continue

                class_name = self._entity_name_from_source(path, text)
                if class_name in self.node_mapping:
                    # Avoid collisions by qualifying duplicate simple names with file stem.
                    class_name = f"{class_name}_{len(self.node_mapping)}"
                self.node_mapping[class_name] = len(self.class_names)
                self.class_names.append(class_name)
                self.file_paths.append(path)
                self.preprocessing_log.append({"file": path, "action": "included", "reason": f"loc:{loc}"})

        if not self.file_paths:
            raise ValueError(
                f"No eligible source files found in {self.root_dir}. "
                f"Lower --min_loc or provide a valid Java project."
            )
        print(f"Found {len(self.file_paths)} eligible Java entities.")

    @staticmethod
    def _architecture_tokens(text: str) -> str:
        """Keep architecture-relevant fragments used by the semantic encoder."""
        lines: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if (
                line.startswith("import ")
                or " class " in f" {line} "
                or " interface " in f" {line} "
                or " enum " in f" {line} "
                or re.search(r"\b(public|private|protected)\b.*\(", line)
                or line.startswith("//")
                or line.startswith("/*")
                or line.startswith("*")
            ):
                lines.append(line)
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
        # High-frequency identifiers act as a deterministic TF-style proxy.
        freq: Dict[str, int] = {}
        for token in identifiers:
            freq[token] = freq.get(token, 0) + 1
        top_ids = [tok for tok, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:128]]
        return "\n".join(lines + top_ids)

    def _hash_embedding(self, text: str) -> torch.Tensor:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
        if not tokens:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, byteorder="little", signed=False)
            idx = raw % self.embedding_dim
            sign = 1.0 if ((raw >> 11) & 1) else -1.0
            vector[idx] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return torch.from_numpy(vector)

    def _get_embedding(self, file_path: str) -> torch.Tensor:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            code = handle.read()
        selected = self._architecture_tokens(code)
        if self.embedding_backend == "codebert" and self.tokenizer is not None and self.bert_model is not None:
            try:
                encoded = self.tokenizer(
                    selected or code,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    output = self.bert_model(**encoded)
                return output.last_hidden_state[0, 0, :].detach().cpu()
            except Exception as exc:
                print(f"Warning: CodeBERT embedding failed for {file_path}: {exc}. Using hashed vector.")
        return self._hash_embedding(selected or code)

    def _regex_dependencies(self, source_name: str, text: str) -> Iterable[Tuple[str, str, str]]:
        """Extract directed dependencies with conservative regex fallbacks."""
        known = set(self.node_mapping.keys())
        # Imports: import a.b.ClassName;
        for imported in re.findall(r"^\s*import\s+([\w\.]+)\s*;", text, flags=re.MULTILINE):
            target = imported.split(".")[-1]
            if target in known and target != source_name:
                yield source_name, target, "import"

        # Extends / implements clauses.
        for target in re.findall(r"\b(?:extends|implements)\s+([A-Z][A-Za-z0-9_]*)", text):
            if target in known and target != source_name:
                yield source_name, target, "inheritance"

        # Object creation and type references.
        for target in re.findall(r"\bnew\s+([A-Z][A-Za-z0-9_]*)\s*\(", text):
            if target in known and target != source_name:
                yield source_name, target, "object_creation"

        # Field/method signatures and local variables using known class names.
        for target in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", text):
            if target in known and target != source_name:
                yield source_name, target, "type_reference"

    def _javalang_dependencies(self, source_name: str, text: str) -> Iterable[Tuple[str, str, str]]:
        if javalang is None:
            yield from self._regex_dependencies(source_name, text)
            return
        try:
            tree = javalang.parse.parse(text)
        except Exception:
            yield from self._regex_dependencies(source_name, text)
            return

        known = set(self.node_mapping.keys())
        for _, node in tree.filter(javalang.tree.Import):
            target = node.path.split(".")[-1] if getattr(node, "path", None) else None
            if target in known and target != source_name:
                yield source_name, target, "import"
        for _, node in tree.filter(javalang.tree.ReferenceType):
            target = getattr(node, "name", None)
            if target in known and target != source_name:
                yield source_name, target, "type_reference"
        for _, node in tree.filter(javalang.tree.ClassCreator):
            type_node = getattr(node, "type", None)
            target = getattr(type_node, "name", None)
            if target in known and target != source_name:
                yield source_name, target, "object_creation"
        for _, node in tree.filter(javalang.tree.ClassDeclaration):
            ext = getattr(node, "extends", None)
            target = getattr(ext, "name", None)
            if target in known and target != source_name:
                yield source_name, target, "inheritance"
            for impl in getattr(node, "implements", []) or []:
                target = getattr(impl, "name", None)
                if target in known and target != source_name:
                    yield source_name, target, "inheritance"

    def build_graph(self) -> SimpleData:
        """Construct a directed attributed dependency graph."""
        if not self.file_paths:
            self.parse_files()

        x_list: List[torch.Tensor] = []
        edges: Dict[Tuple[int, int], str] = {}
        print("Building directed dependency graph...")
        for source_idx, path in enumerate(self.file_paths):
            source_name = self.class_names[source_idx]
            x_list.append(self._get_embedding(path))
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            for src, dst, dep_type in self._javalang_dependencies(source_name, text):
                source = self.node_mapping[src]
                target = self.node_mapping[dst]
                if source != target:
                    edges[(source, target)] = dep_type

        if not x_list:
            raise ValueError("No node embeddings were produced.")
        x = torch.stack(x_list).float()

        if edges:
            sorted_edges = sorted(edges.items())
            edge_pairs = [pair for pair, _ in sorted_edges]
            self.dependency_types = [dep_type for _, dep_type in sorted_edges]
        else:
            print("Warning: no inter-entity dependencies detected; using self-loops for numerical stability.")
            edge_pairs = [(i, i) for i in range(len(x_list))]
            self.dependency_types = ["self_loop"] * len(edge_pairs)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        return SimpleData(
            x=x,
            edge_index=edge_index,
            class_names=self.class_names,
            file_paths=self.file_paths,
            dependency_types=self.dependency_types,
            preprocessing_log=self.preprocessing_log,
        )

    @staticmethod
    def load_ground_truth(gt_path: str) -> Dict[str, int]:
        """Load expert-provided ground truth in Class,True_Module format."""
        ground_truth: Dict[str, int] = {}
        with open(gt_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                name = row.get("Class") or row.get("class") or row.get("Entity")
                module = row.get("True_Module") or row.get("true_module") or row.get("Module")
                if name is not None and module is not None:
                    ground_truth[name.strip()] = int(module)
        return ground_truth
