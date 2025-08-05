import pandas as pd
import numpy as np 
import pywt
from scipy.stats import moment, skew, kurtosis
from scipy.fft import fft
from scipy import signal
import pandas as pd
import numpy as np
from scipy.stats import moment, skew, kurtosis
from scipy.fft import fft
from scipy import signal
import pywt
from statsmodels.tsa.stattools import acf, pacf
import scipy.stats as stats
import matplotlib.pyplot as plt

dataset = pd.read_csv("cleaned_dataset.csv")

def extract_features(row):
    features = {}
    row_values = row.values
    
    features['mean_absolute_deviation'] = np.mean(np.abs(row_values - np.mean(row_values)))
    features['interquartile_range'] = np.percentile(row_values, 75) - np.percentile(row_values, 25)
    
    fft_coeffs = np.abs(fft(row_values))
    features['fft_coefficient_1'] = fft_coeffs[0]
    features['fft_coefficient_2'] = fft_coeffs[1]
    
    coeffs, _ = pywt.dwt(row_values, 'db6')
    features['wavelet_coefficient_1'] = coeffs[0]
    features['wavelet_coefficient_2'] = coeffs[1]
    
    features['I_A^3'] = row['I_A']**3
    features['I_B^3'] = row['I_B']**3
    features['I_C^3'] = row['I_C']**3
    features['I_A^2 * I_B'] = (row['I_A']**2) * row['I_B']
    
    
    features['instantaneous_power'] = row['I_A']**2 + row['I_B']**2 + row['I_C']**2
    
    features['third_moment'] = moment(row_values, moment=3)
    features['fourth_moment'] = moment(row_values, moment=4)
    
    


    
    return pd.Series(features)

total_rows = len(dataset)
for index, row in dataset.iterrows():
    dataset.loc[index, "mean"] = row.mean()
    dataset.loc[index, "median"] = row.median()
    dataset.loc[index, "std"] = row.std()
    dataset.loc[index, "var"] = row.var()
    dataset.loc[index, "min"] = row.min()
    dataset.loc[index, "max"] = row.max()
    dataset.loc[index, "range"] = (row.max()-row.min())
    dataset.loc[index, "skew"] = row.skew()
    dataset.loc[index, "kurtosis"] = row.kurtosis()
    dataset.loc[index, "mode"] = row.mode()[0]  # mode returns a Series, take the first element
    dataset.loc[index, "sum"] = row.sum()
    dataset.loc[index, "i_a-i_b"] = (row['I_A'] - row['I_B'])
    dataset.loc[index, "i_b-i_c"] = (row['I_B'] - row['I_C'])
    dataset.loc[index, "i_a-i_c"] = (row['I_A'] - row['I_C'])
    dataset.loc[index, "i_a*i_b"] = (row['I_A'] * row['I_B'])
    dataset.loc[index, "i_b*i_c"] = (row['I_B'] * row['I_C'])
    dataset.loc[index, "i_a*i_c"] = (row['I_A'] * row['I_C'])
    dataset.loc[index, "i_a*i_b*i_c"] = (row['I_A'] * row['I_B'] * row['I_C'])
    dataset.loc[index, "i_a/i_b"] = (row['I_A'] / row['I_B'])
    dataset.loc[index, "i_b/i_c"] = (row['I_B'] / row['I_C'])
    dataset.loc[index, "i_a/i_c"] = (row['I_A'] / row['I_C'])
    dataset.loc[index, "i_a^2"] = (row['I_A'] * row['I_A'])
    dataset.loc[index, "i_b^2"] = (row['I_B'] * row['I_B'])
    dataset.loc[index, "i_c^2"] = (row['I_C'] * row['I_C'])

    additional_features = extract_features(row)
    for feature, value in additional_features.items():
        dataset.loc[index, feature] = value

    if (index + 1) % 100 == 0 or (index + 1) == total_rows:
        print(f"Processed {index + 1} out of {total_rows} rows ({(index + 1) / total_rows:.2%})")

print("Feature extraction complete.")

dataset.to_csv("featured_dataset.csv", index=False)
