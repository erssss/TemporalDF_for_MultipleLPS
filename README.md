# TemporalDF_for_MultipleLPS

## Directory

│  README.md
│
├─code
│      DF.py
│
└─data
        GPS.csv
        WEATHER.csv

## data

### GPS

- time,sensor1,sensor2,sensor3,sensor4,true

- timestamp(1),observations from different sensors(2~5),true value(6)

### WEATHER

- time,max_temperature_1,max_temperature_2,max_temperature_3,max_temperature_true,min_temperature_1,min_temperature_2,min_temperature_3,min_temperature_true

- date(1), high temperatures from different sources(2~4),true high temperature(5),low temperatures from different sources(6~8),true low temperature(9),
## code

- The function has the same name as the algorithm in the paper.