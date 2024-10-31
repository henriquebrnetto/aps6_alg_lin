import netCDF4, os, re
import pandas as pd
import numpy as np
import autograd.numpy as np_
from autograd import grad

def nc4_to_df(file):

    nc = netCDF4.Dataset(file)
    
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    var = nc.variables['var'][:, :]

    idxs = np.where(~var.mask)

    data = {
        'lat' : lat[idxs[0]],
        'lon' : lon[idxs[1]],
        'var' : var[idxs]
    }
    
    return pd.DataFrame(data)

def sum_var(file):
    nc = netCDF4.Dataset(file)
    var = np.ma.getdata(nc.variables['var'][:, :])

    return np.sum(var[var > 0])

def create_totals_df(folder, pattern=None):
    p = re.compile(pattern)
    years, data = [], []

    for file in os.listdir(folder):
        url = os.path.join(folder, file)
        
        years += [p.search(file).group(1)]
        data += [sum_var(url)]

    return pd.DataFrame(data, index=years, columns=['yield'])

def transform_array(array, n):
    new = array[:n]
    first_val = new[0]
    while True:
        
        array = np.roll(array, -1)
        if array[:n][-1] == first_val:
            break
        new = np.vstack((new, array[:n]))

    return new

def erro( parametros ):
    """
    Funcao retirada da aula 6 notebook
    """
    w, b, x, y_medido = parametros
    yhat = w.T @ x + b
    n = len(x)
    return np_.sum((y_medido - yhat)**2)/ n