import os
import torch
import javalang
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import csv
from collections import defaultdict

class JavaProjectDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.class_names = []
        self.node_mapping = {}
        
        print("Loading CodeBERT model for semantic embeddings...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CodeBERT model: {e}")

    def parse_files(self):
        """Scans directory and collects all .java files with validation."""
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"Path is not a directory: {self.root_dir}")

        idx = 0
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".java"):
                    path = os.path.join(root, file)
                    class_name = file.replace(".java", "")
                    self.file_paths.append(path)
                    self.class_names.append(class_name)
                    self.node_mapping[class_name] = idx
                    idx += 1
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No Java (.java) files found in {self.root_dir}. Please provide a valid Java project.")
        print(f"Found {len(self.file_paths)} classes.")

    def _get_embedding(self, file_path):
        """Generates CodeBERT embedding for a Java file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            if not code.strip():
                print(f"Warning: Empty file {file_path}. Using zero vector.")
                return torch.zeros(768)
            
            tokens = self.tokenizer.tokenize(code)[:510]
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([ids]).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids)
                return outputs.last_hidden_state[0, 0, :].cpu()
        except Exception as e:
            print(f"Warning: Embedding failed for {file_path}: {e}")
            return torch.zeros(768)

    def build_graph(self):
        """Constructs the graph with production-ready fine-grained javalang AST parsing."""
        x_list = []
        edges = set()

        print("Building dependency graph with fine-grained javalang AST parsing...")
        for i, path in enumerate(tqdm(self.file_paths)):
            x_list.append(self._get_embedding(path))

            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if not content.strip():
                    continue
                tree = javalang.parse.parse(content)

                # Fine-grained dependencies for accurate behavior preservation
                # 1. ReferenceType (type usages)
                for _, node in tree.filter(javalang.tree.ReferenceType):
                    if hasattr(node, 'name') and node.name in self.node_mapping:
                        t = self.node_mapping[node.name]
                        if t != i:
                            edges.add((i, t))
                            edges.add((t, i))  # undirected for modularity

                # 2. Import statements
                for _, node in tree.filter(javalang.tree.Import):
                    if node.path and node.path[-1] in self.node_mapping:
                        t = self.node_mapping[node.path[-1]]
                        if t != i:
                            edges.add((i, t))
                            edges.add((t, i))

                # 3. MethodInvocation (call graph)
                for _, node in tree.filter(javalang.tree.MethodInvocation):
                    if node.member in self.node_mapping:
                        t = self.node_mapping[node.member]
                        if t != i:
                            edges.add((i, t))
                            edges.add((t, i))

                # 4. FieldAccess (data dependencies)
                for _, node in tree.filter(javalang.tree.FieldAccess):
                    if hasattr(node, 'member') and node.member in self.node_mapping:
                        t = self.node_mapping[node.member]
                        if t != i:
                            edges.add((i, t))
                            edges.add((t, i))

                # 5. Inheritance / ClassDeclaration (extends/implements)
                for _, node in tree.filter(javalang.tree.ClassDeclaration):
                    if node.extends:
                        for ext in node.extends:
                            if hasattr(ext, 'name') and ext.name in self.node_mapping:
                                t = self.node_mapping[ext.name]
                                if t != i:
                                    edges.add((i, t))
                                    edges.add((t, i))

            except javalang.parser.JavaSyntaxError:
                print(f"Syntax error in {path} - skipping dependencies")
            except Exception as e:
                print(f"Parsing error in {path}: {e}")

        x = torch.stack(x_list)
        
        if not edges:
            print("Warning: No dependencies detected. Adding self-loops.")
            edges = {(i, i) for i in range(len(x_list))}
        
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        
        return Data(
            x=x,
            edge_index=edge_index,
            class_names=self.class_names,
            num_nodes=len(self.class_names),
            file_paths=self.file_paths
        )

    def load_ground_truth(self, gt_path):
        """Load expert-provided ground truth for evaluation."""
        gt = {}
        with open(gt_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    gt[row[0]] = int(row[1])
        return gt