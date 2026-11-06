from .codebook import CHUNK_SIZE, TEXT_TO_VALUE, VALUE_TO_TEXT
from scipy.fft import dct, idct
import numpy as np

def normalize(df, quantiles):
    p1 = quantiles.loc[0.01]
    p99 = quantiles.loc[0.99]
    return (df - p1) / (p99 - p1) * 2 - 1

def denormalize(df, quantiles):
    p1 = quantiles.loc[0.01]
    p99 = quantiles.loc[0.99]
    return (df + 1) / 2 * (p99 - p1) + p1

def quantize(values, scale):
    values = dct(values, norm='ortho')
    values = np.rint(values * scale).astype(int)
    return values

def dequantize(values, scale):
    values = values / scale
    values = idct(values, norm='ortho')
    return values

def to_text(values):
    return ''.join(VALUE_TO_TEXT[v] for v in values)

def from_text(text):
    return np.array([TEXT_TO_VALUE[text[i:i+CHUNK_SIZE]] for i in range(0, len(text), CHUNK_SIZE)])
