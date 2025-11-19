import pandas as pd
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from io import BytesIO

# Charger les modèles
model_soutien_science = joblib.load("model/model_soutien_science.pkl")
model_soutien_litterature = joblib.load("model/model_soutien_litterature.pkl")
model_ressources = joblib.load("model/model_ressources.pkl")
model_orientation = joblib.load("model/model_orientation.pkl")
scaler = joblib.load("model/scalerFinal.pkl")

def importer_fichier_csv(df):
    predictions = []

    for index, row in df.iterrows():
        # Extraire les données de chaque ligne
        ms1, ms2, ms3, ml1, ml2, ml3 = row[['MS1', 'MS2', 'MS3', 'ML1', 'ML2', 'ML3']]
        
        # Utiliser l'index comme ID si la colonne 'id' n'est pas présente
        student_id = row.get('id', index)

        # Préparer les données pour la prédiction
        input_data = np.array([[ms1, ms2, ms3, ml1, ml2, ml3]])
        input_scaled = scaler.transform(input_data)

        # Faire des prédictions
        soutien_science = model_soutien_science.predict(input_scaled[:, :3])
        soutien_litterature = model_soutien_litterature.predict(input_scaled[:, 3:])
        ressources = model_ressources.predict(input_scaled)

        # Identification des matières en échec et en excellence
        echec_sciences = [subj for subj, score in zip(['MS1', 'MS2', 'MS3'], [ms1, ms2, ms3]) if score < 12]
        echec_litterature = [subj for subj, score in zip(['ML1', 'ML2', 'ML3'], [ml1, ml2, ml3]) if score < 12]
        excel_sciences = [subj for subj, score in zip(['MS1', 'MS2', 'MS3'], [ms1, ms2, ms3]) if score >= 15]
        excel_litterature = [subj for subj, score in zip(['ML1', 'ML2', 'ML3'], [ml1, ml2, ml3]) if score >= 15]

        # Déterminer l'orientation scolaire
        if len(echec_sciences) == 3 and len(echec_litterature) == 3:
            orientation_finale = "orientation impossible"
        elif len(excel_sciences) == 3 and len(excel_litterature) == 3:
            orientation_finale = "choix de l'élève"
        else:
            moyenne_sciences = (ms1 + ms2 + ms3) / 3
            moyenne_litterature = (ml1 + ml2 + ml3) / 3
            orientation_finale = "science" if moyenne_sciences > moyenne_litterature else "littéraire"

        # Ajouter les résultats à la liste de prédictions
        predictions.append({
            'ID': student_id, 
            'Soutien Science': 'Besoin de soutien' if soutien_science[0] == 0 else 'Pas de besoin',
            'Matières en Échec Science': ', '.join(echec_sciences),
            'Soutien Littérature': 'Besoin de soutien' if soutien_litterature[0] == 0 else 'Pas de besoin',
            'Matières en Échec Littérature': ', '.join(echec_litterature),
            'Anticipation des Ressources': 'L’élève excelle' if ressources[0] == 'excel' else 'Besoin de ressources supplémentaires',
            'Orientation': orientation_finale
        })

    return predictions


def save_fig_to_bytes(fig):
    """Sauvegarde la figure en mémoire et retourne les bytes."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def graphs(df):
    # Préparer les données pour les prédictions
    X = df[['MS1', 'MS2', 'MS3', 'ML1', 'ML2', 'ML3']]
    X_scaled = scaler.transform(X)

    # Exemple : Créer des vraies étiquettes en fonction des colonnes existantes ou de conditions
    df['true_label_science'] = (df['MS1'] < 12) | (df['MS2'] < 12) | (df['MS3'] < 12)  # Vrai si un soutien est nécessaire
    df['true_label_litterature'] = (df['ML1'] < 12) | (df['ML2'] < 12) | (df['ML3'] < 12)  # Vrai si un soutien est nécessaire

    # Convertir les booléens en int (0 ou 1)
    df['true_label_science'] = df['true_label_science'].astype(int)
    df['true_label_litterature'] = df['true_label_litterature'].astype(int)

    # Prédictions
    y_pred_science = model_soutien_science.predict(X_scaled[:, :3])
    y_pred_litterature = model_soutien_litterature.predict(X_scaled[:, 3:])
    y_pred_orientation = model_orientation.predict(X_scaled)  # Orientation
    # Matrices de confusion
    cm_science = confusion_matrix(df['true_label_science'], y_pred_science)
    cm_litterature = confusion_matrix(df['true_label_litterature'], y_pred_litterature)

    # Graphique pour le besoin de soutien en sciences
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_science, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédiction')
    plt.ylabel('Classe Réelle')
    plt.title("Matrice de Confusion - Sciences")
    plt.text(-0.5, -0.2, '0 = Soutien non requis', fontsize=12, ha='center', va='center')
    plt.text(1.5, -0.2, '1 = Soutien requis', fontsize=12, ha='center', va='center')
    plt.text(0.5, 1.15, "Matière : Sciences", fontsize=12, ha='center', va='center')
    # Enregistrer la figure dans un buffer
    buf_science = save_fig_to_bytes(plt)
    st.download_button(label="Télécharger Heatmap - Sciences", data=buf_science, file_name='heatmap_sciences.png', mime='image/png')
    st.pyplot(plt)
    plt.clf()
    

    # Graphique pour le besoin de soutien en littérature
    lettre = plt.figure(figsize=(8, 6))
    sns.heatmap(cm_litterature, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédiction')
    plt.ylabel('Classe Réelle')
    plt.title("Matrice de Confusion - Littérature")
    plt.text(-0.5, -0.2, '0 = Soutien non requis', fontsize=12, ha='center', va='center')
    plt.text(1.5, -0.2, '1 = Soutien requis', fontsize=12, ha='center', va='center')
    plt.text(0.5, 1.15, "Matière : Littérature", fontsize=12, ha='center', va='center')
    # Enregistrer la figure dans un buffer
    buf_litterature = save_fig_to_bytes(lettre)
    st.download_button(label="Télécharger Heatmap - Littérature", data=buf_litterature, file_name='heatmap_Littérature.png', mime='image/png')
    st.pyplot(lettre)
    plt.clf()

    # Graphique pour l'orientation scolaire
    orien = plt.figure(figsize=(10, 6))
    sns.countplot(x=y_pred_orientation, palette='Set2')
    plt.title('Prédictions d\'Orientation Scolaire')
    plt.xlabel('Orientation (science, littéraire)')
    plt.ylabel('Nombre d\'Étudiants')
    plt.grid(axis='y')
    # Enregistrer la figure dans un buffer
    buf_orien = save_fig_to_bytes(orien)
    st.download_button(label="Télécharger Heatmap - Orientation", data=buf_orien, file_name='heatmap_Orientation.png', mime='image/png')
    st.pyplot(orien)
    plt.clf()


def importe():
    st.title("Prédictions des Besoins Scolaires")

    # Importation de fichier CSV
    uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")

    if uploaded_file is not None:
        # Lire le CSV dans un DataFrame
        df = pd.read_csv(uploaded_file)

        # Vérifier si les colonnes nécessaires sont présentes
        required_columns = ['MS1', 'MS2', 'MS3', 'ML1', 'ML2', 'ML3']
        if all(col in df.columns for col in required_columns):
            # Appeler la fonction importer_fichier_csv avec le DataFrame
            predictions = importer_fichier_csv(df)

            # Afficher les résultats des prédictions sous forme de tableau
            predictions_df = pd.DataFrame(predictions)
            st.subheader("Résultats des Prédictions")
            st.write(predictions_df)

            # Appeler la fonction pour générer et afficher les graphiques
            if st.button("Représentation Graphiques"):
                graphs(df)
        else:
            st.error("Le fichier CSV doit contenir les colonnes MS1, MS2, MS3, ML1, ML2 et ML3.")

if __name__ == "__main__":
    importe()
