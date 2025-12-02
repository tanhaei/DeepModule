import os
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class JavaProjectDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.class_names = []
        self.node_mapping = {} 
        
        # Load CodeBERT model
        print("Loading CodeBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def parse_files(self):
        """Scans the directory for Java files."""
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
        print(f"Found {len(self.file_paths)} classes.")

    def _get_embedding(self, file_path):
        """Helper function to get semantic embedding for a single file using CodeBERT."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            # Tokenize and truncate to fit model limits
            tokens = self.tokenizer.tokenize(code)[:510] 
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(ids).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids)
                # Return the [CLS] token embedding
                return outputs.last_hidden_state[0, 0, :].cpu()
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return torch.zeros(768)

    def build_graph(self):
        """Constructs the PyG Data object (Graph structure + Features)."""
        edge_index = []
        x_list = []

        print("Building Dependency Graph & Extracting Features...")
        for i, path in enumerate(tqdm(self.file_paths)):
            # Feature Extraction (Semantics)
            x_list.append(self._get_embedding(path))

            # Naive Dependency Parsing (Note: Use javalang AST parsing for production)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for other_class, other_idx in self.node_mapping.items():
                        # Simple heuristic: if a class name appears in the file, assume a dependency
                        if i != other_idx and other_class in content:
                            edge_index.append([i, other_idx])
            except:
                pass

        x = torch.stack(x_list)
        if not edge_index: # Handle case with no edges
            edge_index = [[0, 0]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = len(self.class_names)
        data.class_names = self.class_names
        return data