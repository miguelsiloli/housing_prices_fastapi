## Structure
```
housing_algo/
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   └── README.md                # Data description
├── notebooks/                   # Jupyter notebooks for experiments
│   └── data_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data preprocessing scripts
│   ├── model_training.py        # Model training scripts
│   ├── model_inference.py       # Inference script for the trained model
│   └── utils.py                 # Utility functions
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   └── models/                  # Pydantic models for request/response
│       └── prediction.py
├── models/                      # Directory to save trained models
│   └── model.pkl
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_model_inference.py
│   └── test_api.py
├── scripts/                     # Shell scripts for various tasks
│   ├── run_server.sh
│   └── train_model.sh
├── Dockerfile                   # Dockerfile for containerizing the application
├── requirements.txt             # Python dependencies
├── README.md                    # Project description
└── .gitignore                   # Git ignore file
```
