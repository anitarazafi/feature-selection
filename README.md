# feature-selection
Machine Learning project comparing several Feature Selection methods

## Update global configuration:
- raw/processed data paths:
`config/datasets.yaml` 

- results path:
`src/utils/paths.py`

## Local development:
`python3 -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`
`jupyter lab`

## Colab:
- Clone repository:
`!git clone https://github.com/anitarazafi/feature-selection.git`
`%cd feature-selection`
- Install dependencies:
```
import os
if 'requirements.txt' in os.listdir('.'):
    print("requirements.txt found. Proceeding to install dependencies.")
    !pip install -r requirements.txt
else:
    print("requirements.txt not found. No dependencies to install from a file.")
```
