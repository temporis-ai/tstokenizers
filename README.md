# tstokenizer

Time series tokenizer for autoregressive modeling. Converts multivariate time
series into text, then trains a BPE tokenizer on it. Based on the FAST method
by Physical Intelligence.

## Installation

```bash
pip install tstokenizer
```

## Usage

Serialize a DataFrame into text. Its row count must be a multiple of
`patch_len`.

```python
from tstokenizer import Serializer
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'sin': np.sin(np.linspace(0, 10, 128)),
    'cos': np.cos(np.linspace(0, 10, 128)),
})
quantiles = df.quantile([0.01, 0.99])

serializer = Serializer(quantiles, patch_len=8, quant_scale=100)
text = serializer.encode(df)
df_decoded = serializer.decode(text)
patch_text_len = serializer.patch_text_len()
```

Train a BPE tokenizer on a Hugging Face dataset with a `text` column:

```bash
tstokenizer-train --dataset-path=path --output-path=path --vocab-size=1024
```

Encode text into token ids:

```python
from transformers import AutoTokenizer
from tstokenizer import Processor

processor = Processor(AutoTokenizer.from_pretrained(path))
token_ids = processor.encode(text, patch_text_len)
```

## Citation

This package applies the FAST approach to time series:

Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models", 2025.
https://arxiv.org/abs/2501.09747

## License

Apache 2.0
