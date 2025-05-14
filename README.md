# ErePOC

### Environment Setup
```bash
conda env create -f environment.yml
```

### Pretraining
```bash
cd model
python ErePOC_Training.py --train-filepath "(Training Dataset Path).csv" --val-filepath "(Validation Dataset Path).csv"
```

### ErePOC_Tutorial.ipynb
A tutorial notebook for generating and comparing ErePOC pocket representations.

### Seven_ligands_tsne.ipynb
t-SNE visulization of ligand-category-specific pockets using ESM-2 and ErePOC embeddings.