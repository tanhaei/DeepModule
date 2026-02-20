# **DeepModule: Learning to Refactor Software Architectures**

**DeepModule** is an open-source framework based on Graph Neural Networks (GNNs) that optimizes software architecture and suggests microservice candidates by combining structural and semantic analysis of the source code.

## **📂 Project Structure**


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

# DeepModule

## Setup
1. `pip install -r requirements.txt`
2. Generate dummy data for testing: `python generate_dummy_data.py`

## Usage
Run the refactoring pipeline on the example project:
```bash
python main.py --project_dir ./example_project --clusters 
```

### **Installation**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### **Running the Pipeline**

To run the full pipeline (data processing, training, and output generation), use the following command:

```bash
python main.py --project_dir ./path_to_your_java_project
```

Final results will be saved in refactoring_suggestions.csv.
