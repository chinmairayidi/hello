import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from cox_DLL import preprocess_and_predict  # Ensure this matches the filename and function name

# Define the Streamlit app
st.title("Deep Learning Model Predictor")

# Sidebar for CSV upload
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    st.sidebar.markdown("""
[Example input file](https://example.com/example.csv)  # Replace with your actual example file URL
""")

# Collect SMILES strings from the user
smiles_input = st.text_area("Or enter SMILES strings (one per line)", "")

# Button to trigger prediction
if st.button('Predict'):
    if uploaded_file:
        # Handle CSV upload
        try:
            load_data = pd.read_csv(uploaded_file)
            if 'SMILES' not in load_data.columns:
                st.error("The CSV file must contain a column named 'SMILES'.")
            else:
                st.header('**Original input data from CSV**')
                st.write(load_data)

                # Prepare input data for prediction
                drug_data = load_data[['SMILES']]  # Extract only the SMILES column

                try:
                    # Make predictions
                    result_df = preprocess_and_predict(drug_data)

                    # Display results
                    st.write("### Predictions")
                    st.write(result_df)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

    elif smiles_input:
        # Handle SMILES input
        try:
            # Prepare input data
            smiles_list = [smi.strip() for smi in smiles_input.split('\n') if smi.strip()]
            if not smiles_list:
                st.error("No SMILES strings provided.")
            else:
                drug_data = pd.DataFrame({
                    'DrugBank_ID': [f'{i+1}' for i in range(len(smiles_list))],  # Generate unique DrugBank IDs
                    'SMILES': smiles_list
                })

                # Make predictions
                try:
                    result_df = preprocess_and_predict(drug_data)

                    # Display results
                    st.write("### Predictions")
                    st.write(result_df)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        except Exception as e:
            st.error(f"Error processing SMILES input: {e}")

    else:
        st.info("Please upload a CSV file or enter SMILES strings.")
