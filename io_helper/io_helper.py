import numpy as np
import pandas as pd

def load_data(path):
    data = pd.read_csv(path, header=None, sep="	")
    return data.values