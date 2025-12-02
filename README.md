# **DeepModule: Learning to Refactor Software Architectures**

**DeepModule** is an open-source framework based on Graph Neural Networks (GNNs) that optimizes software architecture and suggests microservice candidates by combining structural and semantic analysis of the source code.

## **📂 Project Structure**

```
DeepModule/  
├── src/                     \# Main source code  
│   ├── \_\_init\_\_.py  
│   ├── data\_loader.py      \# Data preprocessing, CodeBERT handling, and graph construction  
│   ├── model.py             \# Neural network architecture (GAT \+ Soft Clustering)  
│   ├── losses.py            \# Loss functions (Modularity, Semantic, Balance)  
│   └── trainer.py           \# Training and evaluation manager  
│  
├── main.py                  \# Main entry point (CLI)  
├── requirements.txt         \# Project dependencies  
└── README.md                \# Documentation
```

## **🚀 Quick Start Guide**

### **Installation**

Install the required dependencies:
```
pip install -r requirements.txt
```

### **Running the Pipeline**

To run the full pipeline (data processing, training, and output generation), use the following command:

```
python main.py \--project\_dir ./path\_to\_your\_java\_project
```

Final results will be saved in refactoring\_suggestions.csv.
