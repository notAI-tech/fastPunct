# fastPunct : Punctuation restoration and spell correction experiments.
[![Downloads](https://pepy.tech/badge/fastpunct)](https://pepy.tech/project/fastpunct)

# Installation:
```bash
pip install --upgrade fastpunct
```

# Supported languages:
english

# Usage:

***As a python module***
```python
from fastpunct import FastPunct
# The default language is 'english'
fastpunct = FastPunct()
fastpunct.punct([
                "john smiths dog is creating a ruccus",
                "ys jagan is the chief minister of andhra pradesh",
                 "we visted new york last year in may"
                 ])
                 
# ["John Smith's dog is creating a ruccus.",
# 'Ys Jagan is the chief minister of Andhra Pradesh.',
# 'We visted New York last year in May.']

# punctuation correction with optional spell correction (experimental)

fastpunct.punct([
                  'johns son peter is marring estella in jun',
                   'kamal hassan is a gud actr'], correct=True)
                   
# ["John's son Peter is marrying Estella in June.",
# 'Kamal Hassan is a good actor.']

```

***As a docker container***
```bash
# Start the docker container
docker run -it -p8080:8080 -eBATCH_SIZE=4 notaitech/fastpunct:english

# Run prediction
curl -d '{"data": ["i was hungry i ordered a pizza my name is batman"]}' -H "Content-Type: application/json" "http://localhost:8080/sync"

# {"prediction": ["I was hungry, I ordered a pizza, my name is Batman."], "success": true}
```

