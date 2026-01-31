

## Data Preparation 

The link to download the raw GEOM Rdkit dataset is [here](https://dataverse.harvard.edu/api/access/datafile/4327252). The file also has the DRUGS dataset, but we don't use that since the dataset is too large for my local memory.

To extract *only* the QM9 dataset, run: 
```
tar -xf 4327252 \
  --wildcards "rdkit_folder/qm9/*" \
  "rdkit_folder/summary_qm9.json"
```

Replace `4327252` with the name of your downloaded file as needed.

Run the preprocessing script to extract and save the data: 

`python preprocess_qm9.py {location of rdkit_folder/}`

## Training

To train the model: 

`python main.py {location of preprocessed_data/}`. See section above. 

## Generating Conformers

To generate conformers, see the notebook: CorrectedConformationViz


