# Databricks Stock Pipeline

A Databricks pipeline for processing and analyzing stock data following the medallion architecture.

## Repository Structure   

```
.
├── notebooks/                  # All pipeline notebooks organized by layer
│   ├── 01_stage/              # Stage layer - raw data ingestion
│   │   └── stage_stocks.ipynb  
│   ├── 02_bronze/             # Bronze layer - initial cleaning
│   │   └── bronze_stocks.ipynb 
│   ├── 03_silver/             # Silver layer - business logic
│   │   └── silver_stocks.ipynb 
│   ├── 04_gold/               # Gold layer - aggregated data
│   │   └── gold_stocks.ipynb   
│   └── 05_predict/            # Prediction layer - ML models
│       └── predict_tomorrow.ipynb
├── pipeline/                   # Pipeline configuration
│   └── stock-pipeline.yml      # Databricks job definition
├── docs/                      # Documentation (optional)
└── tests/                     # Test notebooks (optional)
```

## Pipeline Flow

1. **Stage Layer**: Ingests raw stock data
2. **Bronze Layer**: Cleans and validates the raw data
3. **Silver Layer**: Applies business logic and transformations
4. **Gold Layer**: Creates aggregated datasets for analysis
5. **Predict Layer**: Runs ML models to predict tomorrow's stock prices

## Databricks Setup

1. Import this repository into your Databricks workspace
2. Update the paths in `pipeline/stock-pipeline.yml` if your workspace path differs
3. Create a Databricks job using the YML configuration
4. Set up appropriate cluster configurations

## Deployment

The pipeline is scheduled to run daily at 19 Copenhagen time (see `stock-pipeline.yml`).

## Development

- Add new notebooks to the appropriate layer directory
- Maintain the numbered prefix convention for clarity
- Update the pipeline YML when adding new tasks
