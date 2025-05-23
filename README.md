StrokeSeg-BHSD/
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   ├── processed/
│   │   ├── train/
│   │   └── test/
│   └── logs/
│       └── quality_metrics.csv
│
├── config/
│   └── bhsd.yaml
│
├── preprocessing/
│   ├── preprocess_all.py
│   ├── windowing.py
│   ├── resample_utils.py
│   ├── normalization.py
│   ├── skullstrip.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
