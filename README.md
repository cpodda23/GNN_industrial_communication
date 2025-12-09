# Industrial Wireless Communication with Graph Neural Networks  

It is strongly recommended to use **Python 3.10** due to DGL compatibility issues.

Create and activate a virtual environment:

```bash
python3.10 -m venv gnn_env
source gnn_env/bin/activate
```

## 1. Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Generate the Dataset
```bash
python data_generation.py
```

## 3. Train the Model
```bash
python training.py
```

## 4. Test the Model
```bash
python testing.py
```




