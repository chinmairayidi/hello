import streamlit as st
import pandas as pd
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
                st.write(f"Error: {e}")

    elif smiles_input:
        try:
            # Prepare input data
            smiles_list = [smi.strip() for smi in smiles_input.split('\n') if smi.strip()]
            drug_data = pd.DataFrame({
                'DrugBank_ID': [f'{1}' for i in range(len(smiles_list))],  # Generate dummy DrugBank IDs
                'SMILES': smiles_list
            })

            # Make predictions
            result_df = preprocess_and_predict(drug_data)

            # Display results
            st.write("### Predictions")
            st.write(result_df)
        except Exception as e:
            st.write(f"Error: {e}")

    else:
        st.write("Please upload a CSV file or enter SMILES strings.")



