

- This repository is for paper High Precision $\neq$ High Cost: Temporal Data Fusion for Multiple Low-Precision Sensors.


## File Structure

- code: source code of algorithms.
  - The functions have the same names as the algorithms in the paper.
  - As for TruthFinder, Sums, Investment, CRH, GATD, we use the open source implemenations for them, i.e.,
    - TruthFinder, Sums, Investment: https://github.com/joesingo/truthdiscovery
    - CRH, GATD: https://sites.google.com/iastate.edu/qili
- data: dataset source files used in experiments.

## Dataset

- GPS
  - Manual collection
  - Format: timestamp(1),observations from different sensors(2-5), true value(6)
- WEATHER
  - https://www.aerisweather.com
  - https://www.worldweatheronline.com/
  - https://www.wunderground.com/
- IMU: https://github.com/dusan-nemec/mems-calib
- GINS: https://github.com/i2Nav-WHU/awesome-gins-datasets

## Dependencies

```
numpy==1.24.2
pandas==1.4.3
pyclustering==0.10.1.2
scikit_learn==1.1.1

```


## Instruction

``` sh
cd code
python main.py
```