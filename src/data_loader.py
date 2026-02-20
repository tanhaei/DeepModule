import os
import torch
import javalang
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import csv

class JavaProjectDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.class_names = []
        self.node_mapping = {}
        
        print("Loading CodeBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def parse_files(self):
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
            raise ValueError("No Java files found.")
        print(f"Found {len(self.file_paths)} classes.")

    def _get_embedding(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            if not code.strip():
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
        x_list = []
        edges = set()
        print("Building graph with fine-grained javalang AST parsing...")
        for i, path in enumerate(tqdm(self.file_paths)):
            x_list.append(self._get_embedding(path))
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tree = javalang.parse.parse(content)

                # Fine-grained dependencies
                for _, node in tree.filter(javalang.tree.ReferenceType):
                    if node.name in self.node_mapping:
                        t = self.node_mapping[node.name]
                        if t != i: edges.add((i, t)); edges.add((t, i))
                for _, node in tree.filter(javalang.tree.Import):
                    if node.path and node.path[-1] in self.node_mapping:
                        t = self.node_mapping[node.path[-1]]
                        if t != i: edges.add((i, t)); edges.add((t, i))
                for _, node in tree.filter(javalang.tree.MethodInvocation):
                    if node.member in self.node_mapping:
                        t = self.node_mapping[node.member]
                        if t != i: edges.add((i, t)); edges.add((t, i))
            except Exception:
                pass

        x = torch.stack(x_list)
        if not edges:
            edges = {(i, i) for i in range(len(x_list))}
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, class_names=self.class_names, num_nodes=len(self.class_names))

    def load_ground_truth(self, gt_path):
        gt = {}
        with open(gt_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    gt[row[0]] = int(row[1])
        return gt