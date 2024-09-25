# Electricity consumption forecasting
## Project structure
```bash

.
├── conf
│   ├── model
│   │   ├── conv_gru.yaml
│   │   ├── conv_lin.yaml
│   │   ├── conv_res_like.yaml
│   │   ├── gru.yaml
│   │   └── lstm.yaml
│   ├── config.yaml
│   └── model_registry.yaml
├── .data
│   ├── test.csv.dvc
│   └── train.csv.dvc
├── docker-compose.yml
├── Dockerfile
├── .dvc
│   ├── config
│   └── .gitignore
├── .dvcignore
├── elforecast
│   ├── data
│   │   ├── __init__.py
│   │   ├── datamodule.py
│   │   ├── dataset.py
│   │   └── tools.py
│   └── models
│       ├── __init__.py
│       ├── conv_gru.py
│       ├── conv_lin.py
│       ├── conv_res_like.py
│       ├── gru.py
│       ├── lstm.py
│       └── selector.py
├── images
│   └── prediction_and_usage.png
├── infer.py
├── poetry.lock
├── pyproject.toml
├── README.md
└── train.py
```
## Problem statement
The task is to determine the electricity consumption of a building based on a known date and weather prediction.
Knowing about the energy consumption of a building in the future allows you to get rid of unexpected expenses, as well as implement methods to optimize consumption in advance.

## Data
To begin with, I will use the dataset from [Kaggle](https://www.kaggle.com/competitions/copy-of-challenge23/data).
The data is a csv file with columns:
* Date (date in POSIX format)
* baropressure (data from the weather station)
* humidity (data from the weather station)
* temperature (data from the weather station)
* winddirection (data from the weather station)
* windspeed (data from the weather station)
* ST - energy consumption (data from the electricity meter "Mercury")
* n - the number of meters installed for the specified time period from which energy consumption data was collected

## Models
CNN, ConvResLike, GRU, LSTM, ConvGRU

## The method of prediction
![prediction_and_usage](/images/prediction_and_usage.png)

## How to start
```bash
poetry install
python3 train.py +model={model}
python3 infer.py +model={model}
```

`{model}` - conv_lin, conv_res_like, gru, lstm or conv_gru
