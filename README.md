# **DeepModule: Modularization Recommendation Framework for Java Monoliths**

DeepModule is an **unsupervised** Graph Neural Network framework that recommends architecturally coherent module groupings (candidate microservice boundaries) by combining structural dependencies extracted via production-ready javalang AST parsing and semantic embeddings from CodeBERT.

**Important Scope Clarification** (addressing reviewer feedback):
- The tool **suggests modularization candidates** to support refactoring and microservice identification.
- It does **NOT** perform actual code transformations or guarantee runtime behavior preservation (latency, data consistency, transaction management, etc.).
- All recommendations must be reviewed and implemented by developers.

## 📂 Project Structure


```
DeepModule/  
├── src/                     # Main source code  
│   ├── __init__.py  
│   ├── data_loader.py       # Data preprocessing, CodeBERT handling, and graph construction  
│   ├── model.py             # Neural network architecture (GAT + Soft Clustering)  
│   ├── losses.py            # Loss functions (Modularity, Semantic, Balance)  
│   └── trainer.py           # Training and evaluation manager  
│  
├── generate_dummy_data.py   # Generate dummy data 
├── main.py                  # Main entry point (CLI)  
├── requirements.txt         # Project dependencies  
└── README.md                # Documentation
```

## **🚀 Quick Start Guide**

### **Installation**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### **Generate Example Project**
```bash
python generate_dummy_data.py
```

### **Running the Pipeline**

To run the full pipeline (data processing, training, and output generation), use the following command:

```bash
python main.py --project_dir ./example_project --clusters 3 --epochs 50
```

### **Run with Expert Ground Truth Evaluation (optional)**

```bash
python main.py --project_dir ./example_project --clusters 3 --ground_truth ground_truth.csv
```


Final results will be saved in `modularization_recommendations.csv` and `embeddings.npy` Learned node embeddings (for visualization/qualitative analysis).

