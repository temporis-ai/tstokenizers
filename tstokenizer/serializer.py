from .transforms import *
import pandas as pd

class Serializer:
    def __init__(self, quantiles, patch_len, quant_scale):
        self.quant_scale = quant_scale
        self.patch_len = patch_len
        self.quantiles = quantiles
        self.columns = quantiles.columns
        self.num_columns = len(quantiles.columns)

    def encode(self, df):
        df = normalize(df, self.quantiles)
        values = df[self.columns].to_numpy()
        values = values.reshape(-1, self.patch_len, self.num_columns)
        values = values.transpose(0, 2, 1)
        values = quantize(values, self.quant_scale)
        values = values.transpose(0, 2, 1)
        values = values.reshape(-1)
        text = to_text(values)
        return text

    def decode(self, text):
        values = from_text(text)
        values = values.reshape(-1, self.patch_len, self.num_columns)
        values = values.transpose(0, 2, 1)
        values = dequantize(values, self.quant_scale)
        values = values.transpose(0, 2, 1)
        values = values.reshape(-1, self.num_columns)
        df = pd.DataFrame(values, columns=self.columns)
        df = denormalize(df, self.quantiles)
        return df

    def patch_text_len(self):
        return self.patch_len * self.num_columns * CHUNK_SIZE
