# fastPunct : Fast and accurate punctuation restoration with sequence to sequence networks.
[![Downloads](https://pepy.tech/badge/fastpunct)](https://pepy.tech/project/fastpunct)

# Installation:
```bash
pip install --upgrade fastpunct
```

# Supported languages:
english

# Usage:

```python
from fastpunct import FastPunct
# The default language is 'english'
fastpunct = FastPunct()
fastpunct.punct(["john smiths dog is creating a ruccus", "ys jagan is the chief minister of andhra pradesh", "we visted new york last year in may"])
# ["John Smith's dog is creating a ruccus.", 'Ys Jagan is the chief minister of Andhra Pradesh.', 'We visted New York last year in May.']
```
