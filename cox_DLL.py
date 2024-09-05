#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def preprocess_and_predict(drug_data):
    # Load the trained model
    with open('cnn_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Convert drug_data to DataFrame
    df = pd.DataFrame(drug_data)

    # Convert SMILES to Morgan Fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048) for smi in df['SMILES']]
    fps = np.asarray(fps, dtype=np.int32)

    # Make predictions
    predictions = (model.predict(fps) > 0.3).astype("int32")

    # Combine predictions with DrugBank IDs
    df['Prediction'] = predictions

    return df

