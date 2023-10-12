import pandas as pd
import numpy as np
import os

def convertColName(col_name):
    dist = float(col_name.split("_")[0][:-1])
    if dist == 1.0 or dist == 2.0:
        dist = int(dist)
    angle = float(col_name.split("_")[1][:-3])
    return dist, angle

def linInterpolate(data, num_fills=20, check_for_invalid_data=True):
    d = []
    for i in range(len(data)-1): 
        if check_for_invalid_data and (data[i] == 0 or data[i+1] == 0):
            d.append(np.zeros((num_fills)))
        else:
            d.append(np.linspace(data[i], data[i+1], num_fills))
    return np.array(d).flatten()

def correctMeas(meas, first_meas):
    if first_meas:
        return meas - 0.04 # first measurement
    return meas - 0.005 # second measurement

def loadData(sensor, object, surface, measurement):
    file_name = sensor + '_' + object
    if surface == 'plexiglas':
        file_name += '_plex'  

    if measurement == "first":      
        return pd.read_csv(os.path.join("data", "firstMeasurement", file_name+".csv"))
    elif measurement == "second":
        return pd.read_csv(os.path.join("data", "secondMeasurement", file_name+".csv"))
    elif measurement == "third":
        return pd.read_csv(os.path.join("data", "thirdMeasurement", file_name+".csv"))