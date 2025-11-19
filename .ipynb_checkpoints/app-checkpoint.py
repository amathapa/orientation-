import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle et le scaler
model = joblib.load('modele_knnNew.pkl')
scaler = joblib.load('scalerNew.pkl')

# Titre de l'application
st.title("Prédiction du besoin de soutien")

# Entrée utilisateur pour les notes
notes = st.text_input("Entrez les notes de l'élève (MS1, MS2, MS3) séparées par des virgules :")

# Bouton pour faire la prédiction
if st.button("Prédire"):
    try:
        notes_list = list(map(float, notes.split(',')))
        if len(notes_list) != 3:
            st.error("Veuillez entrer exactement 3 notes.")
        else:
            notes_scaled = scaler.transform([notes_list])
            prediction = model.predict(notes_scaled)
            probabilities = model.predict_proba(notes_scaled)
            # Afficher les résultats
            st.write("Prédiction : '0' pour besoin de soutien, '1' pour pas besoin")
            st.write(prediction[0])
            st.write("Probabilité de besoin de soutien '0' : {:.2f}%".format(probabilities[0][0] * 100))
            st.write("Probabilité de pas besoin de soutien '1' : {:.2f}%".format(probabilities[0][1] * 100))
    except ValueError:
        st.error("Veuillez entrer des valeurs numériques valides.")

# Option pour importer un fichier CSV
uploaded_file = st.file_uploader("Importer un fichier CSV avec les notes des élèves", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Vérifier que le DataFrame contient les colonnes nécessaires
    if all(col in df.columns for col in ['MS1', 'MS2', 'MS3']):
        # Appliquer le scaling et la prédiction pour chaque élève
        notes_scaled = scaler.transform(df[['MS1', 'MS2', 'MS3']])
        predictions = model.predict(notes_scaled)
        probabilities = model.predict_proba(notes_scaled)
        
        # Ajouter les prédictions et la moyenne des probabilités au DataFrame
        df['Prédiction'] = predictions
        df['Probabilités réussite (%)'] = ["{:.2f}%".format(probabilities[i][1] * 100) for i in range(len(probabilities))]  # Conversion en pourcentage

        df['Probabilités échec (%)'] = ["{:.2f}%".format(probabilities[i][0] * 100) for i in range(len(probabilities))]  # Conversion en pourcentage

        # Coloration des cellules et des lignes
        def highlight_cells(row):
            colors = [''] * len(row)  # Initialiser une liste vide de couleurs
            echec_count = 0  # Compteur d'échecs
            for i, note in enumerate([row['MS1'], row['MS2'], row['MS3']]):
                if note < 14:
                    echec_count += 1
                    if note <= 5:
                        colors[i] = 'background-color: rgba(255, 165, 0, 0.5)'  # Rouge foncé
                    else:
                        colors[i] = 'background-color: rgba(255, 165, 0, 0.5)'  # Orange
            if echec_count > 1:  # Colorer toute la ligne si échec dans plus d'une matière
                return ['background-color: rgba(255, 0, 0, 0.5)'] * len(row)  
            return colors

        styled_df = df.style.apply(highlight_cells, axis=1)
         # Afficher les résultats
        st.write("Résultats des prédictions :")
        #st.dataframe(styled_df)
        st.dataframe(df[['MS1', 'MS2', 'MS3', 'Prédiction', 'Probabilités échec (%)', 'Probabilités réussite (%)']].style.apply(highlight_cells, axis=1))
    else:
        st.error("Le fichier doit contenir les colonnes 'MS1', 'MS2' et 'MS3'.")


def plot_histograms(df, seuil_reussite=14):
    plt.figure(figsize=(14, 6))
    subjects = ['MS1', 'MS2', 'MS3']
    
    for i, subject in enumerate(subjects, 1):
        plt.subplot(1, 3, i)
        nombre_eleve = len(df[subject])
        nombre_reussite = len(df[df[subject] >= seuil_reussite])
        pourcentage_reussite = (nombre_reussite / nombre_eleve) * 100 if nombre_eleve > 0 else 0
        
        plt.hist(df[subject], bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=seuil_reussite, color='red', linestyle='--', label='Seuil de réussite (14)')
        plt.title(f'Distribution des notes en {subject}\nPourcentage de réussite: {pourcentage_reussite:.1f}%')
        plt.xlabel('Note')
        plt.ylabel('Nombre d\'élèves')
        plt.legend()
    
    plt.tight_layout()
    st.pyplot(plt)

# Appel de la fonction dans Streamlit
if st.button("Afficher les histogrammes"):
    plot_histograms(df)


def plot_bar_chart(df):
    data = {
        'Matière': ['MS1', 'MS2', 'MS3'],
        'Nombre d\'élèves': [
            len(df[df['MS1'] >= 15]),
            len(df[df['MS2'] >= 15]),
            len(df[df['MS3'] >= 15])
        ]
    }
    bar_df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Matière', y="Nombre d'élèves", data=bar_df, palette='viridis')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', 
                    va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    plt.title("Nombre d'élèves avec des notes >= 15 dans chaque matière")
    plt.xlabel('Matière')
    plt.ylabel("Nombre d'élèves")
    st.pyplot(plt)

# Appel de la fonction dans Streamlit
if st.button("Afficher le graphique à barres"):
    plot_bar_chart(df)


def plot_support_needs(df):
    seuils = {
        'NécesFaible(12-14)': (12, 14),
        'NécesMoyen(10-12)': (10, 12),
        'NécesForte(<10)': (0, 10)
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    matieres = ['MS1', 'MS2', 'MS3']

    for idx, matiere in enumerate(matieres):
        support_counts = {key: 0 for key in seuils.keys()}
        
        for key, (min_val, max_val) in seuils.items():
            if key == 'NécesFaible(12-14)':
                support_counts[key] += len(df[(df[matiere] >= min_val) & (df[matiere] <= max_val)])
            elif key == 'NécesMoyen(10-12)':
                support_counts[key] += len(df[(df[matiere] > min_val) & (df[matiere] < max_val)])
            elif key == 'NécesForte(<10)':
                support_counts[key] += len(df[df[matiere] < max_val])
        
        support_df = pd.DataFrame({
            'Seuil': list(support_counts.keys()),
            'Nombre d\'élèves nécessitant un suivi': list(support_counts.values())
        })

        colors = ['pink', 'lightcoral', 'darkred']
        axes[idx].bar(support_df['Seuil'], support_df['Nombre d\'élèves nécessitant un suivi'], color=colors, edgecolor='black')

        for i, value in enumerate(support_df['Nombre d\'élèves nécessitant un suivi']):
            axes[idx].text(i, value + 1, str(value), ha='center', va='bottom')

        axes[idx].set_title(f'Nombre d\'élèves nécessitant un suivi en {matiere}')
        axes[idx].set_xlabel('Seuil')
        axes[idx].set_ylabel('Nombre d\'élèves')

    plt.tight_layout()
    st.pyplot(plt)

# Appel de la fonction dans Streamlit
if st.button("Afficher le suivi des besoins"):
    plot_support_needs(df)





##############################################


import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Masquer les options de Streamlit
st.set_page_config(page_title="Predictor.ndye", page_icon=":guardsman:", layout="wide")

# Charger le modèle et le scaler
model = joblib.load('../models/modele_knnNew.pkl')
scaler = joblib.load('../models/scalerNew.pkl')

# Titre de l'application
st.title("Prédiction du besoin de soutien")

# Entrée utilisateur pour les notes
notes = st.text_input("Entrez les notes de l'élève (MS1, MS2, MS3) séparées par des virgules :")

# Bouton pour faire la prédiction
if st.button("Prédire"):
    try:
        notes_list = list(map(float, notes.split(',')))
        if len(notes_list) != 3:
            st.error("Veuillez entrer exactement 3 notes.")
        else:
            # Appliquer le scaler et faire la prédiction
            notes_scaled = scaler.transform([notes_list])
            prediction = model.predict(notes_scaled)
            probabilities = model.predict_proba(notes_scaled)

            # Afficher les résultats
            st.write("Prédiction : '0' pour besoin de soutien, '1' pour pas besoin")
            st.write(prediction[0])
            st.write("Probabilité de besoin de soutien '0' : {:.2f}%".format(probabilities[0][0] * 100))
            st.write("Probabilité de pas besoin de soutien '1' : {:.2f}%".format(probabilities[0][1] * 100))
    except ValueError:
        st.error("Veuillez entrer des valeurs numériques valides.")

# Option pour importer un fichier CSV
uploaded_file = st.file_uploader("Importer un fichier CSV avec les notes des élèves", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Vérifier que le DataFrame contient les colonnes nécessaires
    if all(col in df.columns for col in ['MS1', 'MS2', 'MS3']):
        # Appliquer le scaling et la prédiction pour chaque élève
        notes_scaled = scaler.transform(df[['MS1', 'MS2', 'MS3']])
        predictions = model.predict(notes_scaled)
        probabilities = model.predict_proba(notes_scaled)

        # Ajouter les prédictions et les probabilités au DataFrame
        df['Prédiction'] = predictions
        df['Probabilités réussite (%)'] = ["{:.2f}%".format(probabilities[i][1] * 100) for i in range(len(probabilities))]
        df['Probabilités échec (%)'] = ["{:.2f}%".format(probabilities[i][0] * 100) for i in range(len(probabilities))]

        # Coloration des cellules et des lignes
        def highlight_cells(row):
            colors = [''] * len(row)
            echec_count = 0
            for i, note in enumerate([row['MS1'], row['MS2'], row['MS3']]):
                if note < 14:
                    echec_count += 1
                    colors[i] = 'background-color: rgba(255, 165, 0, 0.5)' if note > 5 else 'background-color: rgba(255, 0, 0, 0.5)'
                if echec_count > 1:
                    return ['background-color: rgba(255, 0, 0, 0.5)'] * len(row)
            return colors

        styled_df = df.style.apply(highlight_cells, axis=1)
        st.write("Résultats des prédictions :")
        st.dataframe(df[['MS1', 'MS2', 'MS3', 'Prédiction', 'Probabilités échec (%)', 'Probabilités réussite (%)']].style.apply(highlight_cells, axis=1))
    else:
        st.error("Le fichier doit contenir les colonnes 'MS1', 'MS2' et 'MS3'.")

    # Afficher les histogrammes
    def plot_histograms(df, seuil_reussite=14):
        plt.figure(figsize=(14, 6))
        subjects = ['MS1', 'MS2', 'MS3']

        for i, subject in enumerate(subjects, 1):
            plt.subplot(1, 3, i)
            plt.hist(df[subject], bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(x=seuil_reussite, color='red', linestyle='--', label='Seuil de réussite (14)')
            plt.title(f'Distribution des notes en {subject}')
            plt.xlabel('Note')
            plt.ylabel('Nombre d\'élèves')
            plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    if st.button("Afficher les histogrammes"):
        plot_histograms(df)

    # Graphique à barres des résultats
    def plot_bar_chart(df):
        data = {
            'Matière': ['MS1', 'MS2', 'MS3'],
            'Nombre d\'élèves': [
                len(df[df['MS1'] >= 15]),
                len(df[df['MS2'] >= 15]),
                len(df[df['MS3'] >= 15])
            ]
        }
        bar_df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Matière', y="Nombre d'élèves", data=bar_df, palette='viridis')

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', 
                        va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')

        plt.title("Nombre d'élèves avec des notes >= 15 dans chaque matière")
        plt.xlabel('Matière')
        plt.ylabel("Nombre d'élèves")
        st.pyplot(plt)

    if st.button("Afficher le graphique à barres"):
        plot_bar_chart(df)

    # Suivi des besoins
    def plot_support_needs(df):
        seuils = {
            'NécesFaible(12-14)': (12, 14),
            'NécesMoyen(10-12)': (10, 12),
            'NécesForte(<10)': (0, 10)
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        matieres = ['MS1', 'MS2', 'MS3']

        for idx, matiere in enumerate(matieres):
            support_counts = {key: 0 for key in seuils.keys()}
            for key, (min_val, max_val) in seuils.items():
                support_counts[key] += len(df[(df[matiere] >= min_val) & (df[matiere] <= max_val)]) if key == 'NécesFaible(12-14)' else len(df[(df[matiere] > min_val) & (df[matiere] < max_val)]) if key == 'NécesMoyen(10-12)' else len(df[df[matiere] < max_val])
            
            support_df = pd.DataFrame({
                'Seuil': list(support_counts.keys()),
                'Nombre d\'élèves nécessitant un suivi': list(support_counts.values())
            })

            colors = ['pink', 'lightcoral', 'darkred']
            axes[idx].bar(support_df['Seuil'], support_df['Nombre d\'élèves nécessitant un suivi'], color=colors, edgecolor='black')

            for i, value in enumerate(support_df['Nombre d\'élèves nécessitant un suivi']):
                axes[idx].text(i, value + 1, str(value), ha='center', va='bottom')

            axes[idx].set_title(f'Nombre d\'élèves nécessitant un suivi en {matiere}')
            axes[idx].set_xlabel('Seuil')
            axes[idx].set_ylabel('Nombre d\'élèves')

        plt.tight_layout()
        st.pyplot(plt)

    if st.button("Afficher le suivi des besoins"):
        plot_support_needs(df)
