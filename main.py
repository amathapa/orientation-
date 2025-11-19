import streamlit as st
from app2 import run_predictions
from import_csv import importer_fichier_csv, importe

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Prédictions", "Importer CSV"])
    if page == "Prédictions":
        run_predictions()
        
    elif page == "Importer CSV":
        importe()

if __name__ == "__main__":
    main()
