# ppi-with-stacked-autoencoders

Implementation of SAE from [Sequence-based prediction of protein protein interaction using a deep learning algorithm](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1700-2) (Sun et al. 2017) in PyTorch.

## Data

FASTA sequences available in `data/human.fasta` and `data/yeast.fasta`. Labels available in `data/negativeSimTest.zip`. The FASTA files encoded using the Autocovariance method and Conjoint Triad method are stored as .npy files in `data/human_ac.zip`, `data/human_ct.zip`, `data/yeast_ac.zip`, `data/yeast_ct.zip`. For the human data, the lag in the Autocovariance method is 20, and for yeast, it is 30. Hence, human AC vectors are dim 140 and yeast AC vectors are dim 210. The model input size for human AC is 280 and is 420 for yeast. The model input size for CT is 686 for both human and yeast data.

## Training

See the options in `train.py`. An example for training SAE on human AC data. Default is 100 epochs and batch size of 100.
```python
python train.py --model classifier --data-type human --data-dir data/human_ac --feature-type AC --debug False --num-folds-to-use 10 --input-size 280 --hidden-layer-size 400 --lr 0.01
```
