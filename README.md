# Machine Learning for Peptide Synthesis

This is a machine learning project to predict peptide synthesis outcomes. It uses data from UV absorption following Fmoc deprotections to infer coupling efficiency. This data is then preprocessed and used to train a variety of ML models and their accuracy is compared using cross-validation and other techniques. Running the script will generate some comparison charts.

Absoprtion data is noisy, so I have classified the range of first differentials into 5 categories to better guide the ML models.

Please note: This project uses data from the following: Bálint Tamás, Marvin Alberts, Teodoro Laino, and Nina Hartrampf, (2025). Raw Experimental Data for "Amino Acid Composition Drives Peptide Aggregation: Predicting Aggregation for Improved Synthesis." Zenodo. https://doi.org/10.26434/chemrxiv-2025-wjbmv
https://zenodo.org/records/14824562 

Please note: I **DID NOT** collect the data myself nor did contribute to the associated manuscript in *any way*. I am simply using their data under creative commons licence for my own project. 

## Installation 

The following instructions use pip and python virtual environments. If you wish to use an alternative package/environment manager (like conda), I am sure the steps are similiar. 


 
