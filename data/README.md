## Autocovariance method

See `yeast_ac/ignore.txt` for list of proteins that will be excluded because length is less than `lag=30`.
See `human_ac/ignore.txt` for list of proteins that will be excluded because length is less than `lag=20`. Proteins with `U` and `X` amino acids are also excluded.

* Yeast/Human AC - the physicochemical table (see https://github.com/pemami4911/ppi-with-stacked-autoencoders/blob/master/data_preprocessing.ipynb) is normalized column-wise to be mean zero gaussian
* Yeast CT - frequency values are normalized by dividing through by 29 (95% of max frq values are less than 29). 
* Human CT - frequency values are normalized by dividing through by 33 (" " ")
