# ppi-with-stacked-autoencoders

Implementation of SAE from [Sequence-based prediction of protein protein interaction using a deep learning algorithm](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1700-2) (Sun et al. 2017) in PyTorch.

## Data
This is based on the Human and Yeast data used in the Profppikernel paper [[see here]](https://rostlab.org/owiki/index.php/More_challenges_for_the_prediction_of_protein-protein_interactions). We use C1 for training and C1+C2 test sets for validation in our implementation.

FASTA sequences available in `data/human.fasta` and `data/yeast.fasta`. Labels available in `data/negativeSimTest.zip`. The FASTA files encoded using the Autocovariance (AC) method and Conjoint Triad (CT) method are stored as .npy files in `data/human_ac.zip`, `data/human_ct.zip`, `data/yeast_ac.zip`, `data/yeast_ct.zip`. For the human data, the lag in the Autocovariance method is 20, and for yeast, it is 30. Hence, human AC vectors are dim 140 and yeast AC vectors are dim 210. The model input size for human AC is 280 and is 420 for yeast. The model input size for CT is 686 for both human and yeast data.

## Autocovariance and conjoint triad

These algorithms for featurizing FASTA sequences are implemented in the [data preprocessing notebook](https://github.com/pemami4911/ppi-with-stacked-autoencoders/blob/master/data_preprocessing.ipynb).

## Hyperparameters

For AC, use 400 hidden units (--hidden-layer-size) and for CT, use 700 hidden units. We had good results with the Adam optimizer (default) with learning rate 0.01. We used batch size of 100 and trained for 100 epochs. Results are averaged over 10 fold cross-validation.

## Training

See the options in `train.py`. An example for training SAE on human AC data. Default is 100 epochs and batch size of 100.
```python
python train.py --model classifier --data-type human --data-dir data/human_ac --feature-type AC --debug False --num-folds-to-use 10 --input-size 280 --hidden-layer-size 400 --lr 0.01
```

## Performance

We got ~62% and ~60% validation prediction accuracy averaged over 10 folds for human and yeast datasets, respectively. 
