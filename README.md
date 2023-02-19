# Premium cars classification

Simple classification project based on car sale adverts, published on German version of eBay in years 2014-2016.

# Project's description

This project's goal is to classify two groups of registered in XXI Century cars' brands - premium and non-premium - basing on features like engine power, vehicle type, year of 1st registration, engine fuel type etc.

# Technologies used

- The file is written in Python (version 3.10.6).
- The algorithm used to classify is XGBClassifier, which is characterized by ability to improve after each iteration and good working on imbalanced sets.
In notebook you can also find the tries of modelling by ComplementNB, BalancedRFC, MLPClassifier or KNeighbors Classifier and the comparison of algorithms above to selected one.
- Other used libraries are described in requirements.

# Installation

Write the command below:
```bash
python setup.py install
```
or
```bash
pip install -r requirements.txt
```

in command prompt.

# License

MIT.

# Authors

Jakub Walczak (https://github.com/jakubtwalczak), 
Szymon Łątkowski (https://github.com/slatkowski)
