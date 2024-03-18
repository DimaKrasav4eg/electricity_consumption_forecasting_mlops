# Electricity consumption forecasting
## Project structure
```bash

.
├── docker-compose.yml
├── Dockerfile
├── elforecast
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── tools.py
│   └── models
│       └── __init__.py
│       ├── conv_lstm.py
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

## Model
I will use the CNN-LSTM model
* optimezer - Adam
* loss-func - MAE
## The method of prediction
![prediction_and_usage](/images/prediction_and_usage.png)