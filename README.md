# fastPunct : Fast and accurate punctuation restoration with sequence to sequence networks.

# Installation:
```bash
pip install --upgrade fastpunct
```

# Supported languages:
en - english

# Usage:

```python
from fastpunct import FastPunct
# The default language is 'en'
fastpunct = FastPunct('en')
fastpunct.punct(["oh i thought you were here", "in theory everyone knows what a comma is", "hey how are you doing", "my name is sheela i am in love with hrithik"], batch_size=32)
# ['Oh! I thought you were here.', 'In theory, everyone knows what a comma is.', 'Hey! How are you doing?', 'My name is Sheela. I am in love with Hrithik.']

```