import spidev
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib  # Import joblib for model loading
from sklearn.tree import DecisionTreeClassifier
import pickle
from scipy.fft import fft
from scipy.stats import moment, mode, skew, kurtosis
import pywt
from statsmodels.tsa.stattools import acf, pacf


def setup_spi(channel=0, device=0, speed=3600000, mode=0):
    spi = spidev.SpiDev()
    spi.open(channel, device)
    spi.max_speed_hz = speed
    spi.mode = mode
    return spi


def load_model(model_path):
    try:
        print("Loading model")
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_features(row):
    features = {}
    row_values = row.values

    features['mean']   = np.mean(row_values)
    features['median'] = np.median(row_values)
    features['std']    = np.std(row_values, ddof=1)  # Sample std
    features['var']    = np.var(row_values, ddof=1)  # Sample var
    features['min']    = np.min(row_values)
    features['max']    = np.max(row_values)
    features['range']  = features['max'] - features['min']
    features['sum']    = np.sum(row_values)

    features['skew']     = skew(row_values)       # By default, bias=False
    features['kurtosis'] = kurtosis(row_values)   # By default, fisher=True

    mode_result = mode(row_values, keepdims=True)
    features['mode'] = mode_result.mode[0]

    features['i_a-i_b'] = row['I_A'] - row['I_B']
    features['i_b-i_c'] = row['I_B'] - row['I_C']
    features['i_a-i_c'] = row['I_A'] - row['I_C']

    features['i_a*i_b']     = row['I_A'] * row['I_B']
    features['i_b*i_c']     = row['I_B'] * row['I_C']
    features['i_a*i_c']     = row['I_A'] * row['I_C']
    features['i_a*i_b*i_c'] = row['I_A'] * row['I_B'] * row['I_C']

    features['i_a/i_b'] = (row['I_A'] / row['I_B']) if row['I_B'] != 0 else np.nan
    features['i_b/i_c'] = (row['I_B'] / row['I_C']) if row['I_C'] != 0 else np.nan
    features['i_a/i_c'] = (row['I_A'] / row['I_C']) if row['I_C'] != 0 else np.nan

    features['i_a^2'] = row['I_A']**2
    features['i_b^2'] = row['I_B']**2
    features['i_c^2'] = row['I_C']**2
    
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
    
    features['third_moment']  = moment(row_values, moment=3)
    features['fourth_moment'] = moment(row_values, moment=4)
    
    


    

    
    return pd.Series(features)


def preprocess_data(data):
    
    df = pd.DataFrame([{'I_A': data[0], 'I_B': data[1], 'I_C': data[2]}])
    
    
    features_series = extract_features(df.iloc[0])
    
    features_array = features_series.values.reshape(1, -1)
    
    return features_array


def read_adc(spi, channel):
    if channel < 0 or channel > 7:
        raise ValueError("ADC channel must be between 0 and 7.")

    command = [1, (8 + channel) << 4, 0]
    response = spi.xfer2(command)
    adc_value = ((response[1] & 3) << 8) + response[2]
    offset_val = 238
    adc_value = adc_value / 1023 * 500
    adc_value = adc_value - offset_val
    return adc_value


def initialize_plot(xlen=50, y_range=[0, 500]):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    xs = list(range(0, xlen))
    
    ys_left = {0: [0] * xlen, 1: [0] * xlen, 2: [0] * xlen}
    ys_right = {3: [0] * xlen, 4: [0] * xlen, 5: [0] * xlen}
    
    lines_left = [ax.plot(xs, ys_left[ch], label=f'Ch {ch}')[0] for ch in range(3)]
    lines_right = [ax2.plot(xs, ys_right[ch], label=f'Ch {ch}')[0] for ch in range(3, 6)]
    
    ax.set_ylabel('Voltage (mV)')
    ax.set_xlabel('Samples')
    ax.set_ylim(-max(y_range), max(y_range))
    ax2.set_ylabel('Current (mA)')
    ax2.set_xlabel('Samples')
    ax2.set_ylim(-max(y_range), max(y_range))
    
    plt.tight_layout()
    
    return fig, ax, ax2, xs, ys_left, ys_right, lines_left, lines_right


def update_data(ys_left, ys_right, xlen, spi):
    for channel in range(0, 3):
        val = read_adc(spi, channel)
        ys_left[channel].append(val)

    for channel in range(0, 3):
        ys_left[channel] = ys_left[channel][-xlen:]

    for channel in range(3, 6):
        val = read_adc(spi, channel)
        ys_right[channel].append(val)

    for channel in range(3, 6):
        ys_right[channel] = ys_right[channel][-xlen:]

    return ys_left, ys_right


def run_inference(model, ys_left, ys_right):
    input_data = np.array([ys_left[ch][-1] for ch in range(3)])
 
    processed_input = preprocess_data(input_data)
    
    prediction = model.predict(processed_input)
    
    return prediction



rolling_buffer = []      # Will store dicts of {'time': ..., 'I_A': ..., 'I_B': ..., 'I_C': ...}
last_inference_time = 0  # Track when we last performed inference

def build_single_row_for_inference(df_window):
    i_a_mean = df_window['I_A'].mean()
    i_a_std  = df_window['I_A'].std() if len(df_window) > 1 else 0.0  # if only one sample, std=0

    last_sample = df_window.iloc[-1]  # has 'I_A', 'I_B', 'I_C', 'time'
    
    row_dict = {
        'I_A': last_sample['I_A'],
        'I_B': last_sample['I_B'],
        'I_C': last_sample['I_C']
    }

    return pd.DataFrame([row_dict])


def animate(frame_idx, ys_left, ys_right, lines_left, lines_right, xlen, spi, model):
    global rolling_buffer, last_inference_time
    
    current_time = time.time()
    
    ys_left, ys_right = update_data(ys_left, ys_right, xlen, spi)
    
    i_a = ys_left[0][-1]
    i_b = ys_left[1][-1]
    i_c = ys_left[2][-1]

    rolling_buffer.append({
        'time': current_time,
        'I_A':  i_a,
        'I_B':  i_b,
        'I_C':  i_c
    })
    
    one_second_ago = current_time - 1.0
    rolling_buffer = [row for row in rolling_buffer if row['time'] >= one_second_ago]
    
    if (current_time - last_inference_time) >= 1.0:
        df_window = pd.DataFrame(rolling_buffer)
        if len(df_window) > 0:
            single_row_df = build_single_row_for_inference(df_window)
            
            feats = extract_features(single_row_df.iloc[0])  # .iloc[0] => a Series
            feat_array = feats.values.reshape(1, -1)
            
            prediction = model.predict(feat_array)
            print(f"[{time.strftime('%H:%M:%S')}] Prediction:", prediction)
        else:
            print("No data in the last 1 second. Skipping inference.")
        
        last_inference_time = current_time
    
    for ch in range(0, 3):
        lines_left[ch].set_ydata(ys_left[ch])
    for ch in range(3, 6):
        lines_right[ch - 3].set_ydata(ys_right[ch])
    
    return lines_left + lines_right


def main():
    spi = setup_spi()

    model_path = 'Models/pkl/DecisionTree.pkl'  # Path to the model saved as .pth
    model = load_model(model_path)
    
    if model is None:
        print("Failed to load the model. Exiting...")
        return

    xlen = 50
    y_range = [0, 500]
    fig, ax, ax2, xs, ys_left, ys_right, lines_left, lines_right = initialize_plot(xlen, y_range)

    ani = animation.FuncAnimation(fig, animate, fargs=(ys_left, ys_right, lines_left, lines_right, xlen, spi, model), interval=28.5, blit=False)

    plt.show()

    spi.close()


if __name__ == "__main__":
    main()
