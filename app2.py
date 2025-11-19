import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

model_soutien_science = joblib.load("model/model_soutien_science.pkl")
model_soutien_litterature = joblib.load("model/model_soutien_litterature.pkl")
model_ressources = joblib.load("model/model_ressources.pkl")
model_orientation = joblib.load("model/model_orientation.pkl")
scaler = joblib.load("model/scalerFinal.pkl")

st.title("Prédiction des besoins étudiants")

def run_predictions():
    st.header("Entrez les notes des étudiants")

    ms1 = st.number_input("Note Math", min_value=0, max_value=20, key="ms1")
    ms2 = st.number_input("Note P/C", min_value=0, max_value=20, key="ms2")
    ms3 = st.number_input("Note Biologie", min_value=0, max_value=20, key="ms3")
    ml1 = st.number_input("Note Français", min_value=0, max_value=20)
    ml2 = st.number_input("Note Anglais", min_value=0, max_value=20)
    ml3 = st.number_input("Note Philosophie", min_value=0, max_value=20)
    

    if st.button("Prédire"):
                # Vérification si tous les champs sont renseignés
        if ms1 == 0 and ms2 == 0 and ms3 == 0 and ml1 == 0 and ml2 == 0 and ml3 == 0:
            st.error("Veuillez entrer les notes pour au moins un étudiant.")
        else:
            input_data = np.array([[ms1, ms2, ms3, ml1, ml2, ml3]])
            input_scaled = scaler.transform(input_data)

            soutien_science = model_soutien_science.predict(input_scaled[:, :3])
            soutien_litterature = model_soutien_litterature.predict(input_scaled[:, 3:])
            ressources = model_ressources.predict(input_scaled)

            # Calculer les probabilités pour l'orientation
            orientation_probabilities = model_orientation.predict_proba(input_scaled)  # Retourne un tableau 2D
            orientation_labels = model_orientation.classes_  # Récupérer les étiquettes des classes
            orientation_percentages = {label: prob for label, prob in zip(orientation_labels, orientation_probabilities[0])}

            echec_sciences, echec_litterature = [], []
            excel_sciences, excel_litterature = [], []

            st.subheader("Résultats des prédictions")

            if soutien_science[0] == 0:
                st.write("Sciences : **Soutien Requis** ❌")
                if ms1 < 13:
                    echec_sciences.append("Math")
                if ms2 < 13:
                    echec_sciences.append("P/C")
                if ms3 < 13:
                    echec_sciences.append("Biologie")
                st.write(f"Matières en soutien : {', '.join(echec_sciences)}")
            else:
                st.write("Sciences : **Soutien non Requis** ✔️")
                if ms1 >= 16:
                    excel_sciences.append("Math")
                if ms2 >= 16:
                    excel_sciences.append("P/C")
                if ms3 >= 16:
                    excel_sciences.append("Biologie")
                st.write(f"Matières dominante : {', '.join(excel_sciences)}")


            if soutien_litterature[0] == 0:
                st.write("Littérature : **Soutien Requis** ❌")
                if ml1 < 13:
                    echec_litterature.append("Français")
                if ml2 < 13:
                    echec_litterature.append("Anglais")
                if ml3 < 13:
                    echec_litterature.append("Philosophie")
                st.write(f"Matières en soutien : {', '.join(echec_litterature)}")
            else:
                st.write("Littérature : **Soutien non Requis** ✔️")
                if ml1 >= 16:
                    excel_litterature.append("Français")
                if ml2 >= 16:
                    excel_litterature.append("Anglais")
                if ml3 >= 16:
                    excel_litterature.append("Philosophie")
                st.write(f"Matières dominante : {', '.join(excel_litterature)}")


            if ressources[0] == 'excel':
                st.write("Matières dans lesquelles l'élève excelle : ")
                excel_matieres = []
                if ms1 >= 16:
                    excel_matieres.append("Math")
                if ms2 >= 16:
                    excel_matieres.append("P/C")
                if ms3 >= 16:
                    excel_matieres.append("Biologie")
                if ml1 >= 16:
                    excel_matieres.append("Français")
                if ml2 >= 16:
                    excel_matieres.append("Anglais")
                if ml3 >= 16:
                    excel_matieres.append("Philosophie")
                st.write(f"{', '.join(excel_matieres)} 🎉")
            else:
                st.write("Matières nécessitant des ressources supplémentaires : ")
                besoin_matieres = []
                if ms1 < 13:
                    besoin_matieres.append("Math")
                if ms2 < 13:
                    besoin_matieres.append("P/C")
                if ms3 < 13:
                    besoin_matieres.append("Biologie")
                if ml1 < 13:
                    besoin_matieres.append("Français")
                if ml2 < 13:
                    besoin_matieres.append("Anglais")
                if ml3 < 13:
                    besoin_matieres.append("Philosophie")
                st.write(f" {', '.join(besoin_matieres)} ⚠️")

            moyenne_sciences = (ms1 + ms2 + ms3) / 3
            moyenne_litterature = (ml1 + ml2 + ml3) / 3

            if len(echec_sciences) == 3 and len(echec_litterature) == 3:
                orientation_finale = "en attente de soutien"
            elif len(excel_sciences) == 3 and len(excel_litterature) == 3:
                orientation_finale = "choix de l'élève"
            else:
                if moyenne_sciences > moyenne_litterature:
                    orientation_finale = "science"
                else:
                    orientation_finale = "littéraire"

                    # Afficher les probabilités d'orientation
            st.subheader("Probabilités d'Orientation")
            for label, prob in orientation_percentages.items():
                st.write(f"{label} : {prob * 100:.2f}%")
            st.write(f"Orientation scolaire : {orientation_finale}")



run_predictions()
