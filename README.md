# Industrial Wireless Communication with Graph Neural Networks  
Minimal Installation & Run Instructions

## 1. Requirements
It is strongly recommended to use **Python 3.10** due to DGL compatibility issues.

Create and activate a virtual environment:

```bash
python3.10 -m venv gnn_env
source gnn_env/bin/activate
```

## Install dependencies:
pip install -r requirements.txt

## 1. Generate the Dataset
python data_generation.py

## 3. Train the Model
python training.py

## 4. Test the Model
python testing.py




