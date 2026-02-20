import os
import torch
import javalang
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

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
        """Scans the directory and validates Java files."""
        idx = 0
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

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
            raise ValueError("No Java files found in the project directory.")
        print(f"Found {len(self.file_paths)} classes.")

    def _get_embedding(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            tokens = self.tokenizer.tokenize(code)[:510] 
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(ids).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids)
                return outputs.last_hidden_state[0, 0, :].cpu()
        except Exception as e:
            print(f"Warning: Failed to embed {file_path}. Using zero vector. Error: {e}")
            return torch.zeros(768)

    def build_graph(self):
        """Constructs the graph using AST-based dependency analysis."""
        edge_index = []
        x_list = []

        print("Building Dependency Graph via AST Parsing...")
        for i, path in enumerate(tqdm(self.file_paths)):
            x_list.append(self._get_embedding(path))

            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    tree = javalang.parse.parse(content)
                
                # Extract dependencies using AST nodes
                for _, node in tree.filter(javalang.tree.ReferenceType):
                    if node.name in self.node_mapping:
                        edge_index.append([i, self.node_mapping[node.name]])
                        
            except Exception as e:
                print(f"Parsing error in {path}: {e}")

        x = torch.stack(x_list)
        if not edge_index:
            edge_index = [[i, i] for i in range(len(x_list))] # Self-loops if no edges
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, class_names=self.class_names, num_nodes=len(self.class_names))