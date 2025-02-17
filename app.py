import streamlit as st
import joblib
import numpy as np

# Charger le modèle de régression logistique
logreg_model = joblib.load('logreg_model3.pkl')

# Fonction de prédiction
def predict_churn(gender, senior_citizen, tenure, contract, payment_method, 
                  monthly_charges, total_charges, threshold=0.5):  
    
    input_data = np.array([float(gender), float(senior_citizen), float(tenure), float(contract),  
                       float(payment_method), float(monthly_charges), float(total_charges)]).reshape(1, -1)

    churn_probability = logreg_model.predict_proba(input_data)[:, 1][0]  
    churn_prediction = 1 if churn_probability >= threshold else 0
    
    return churn_prediction, churn_probability

# === Interface Streamlit ===
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Onglet de navigation
menu = st.sidebar.radio("Navigation", ["Accueil", "Voir Dashboard", "Voir Rapport"])

if menu == "Accueil":
    st.title('Customer Churn Prediction')
    st.write("""
        Cette application prédit la probabilité qu'un client quitte l'entreprise (churn) en fonction de ses détails.
        Entrez les informations du client ci-dessous et obtenez la prédiction.
    """)

    # Seuil ajustable
    threshold = st.slider('Sélectionner le seuil de prédiction de churn', 0.0, 1.0, 0.5)

    # Entrées utilisateur
    gender = st.selectbox('Gender', ['Male', 'Female'])
    gender = 1 if gender == 'Female' else 0

    senior_citizen = st.selectbox('Senior Citizen (1: Yes, 0: No)', [0, 1])
    tenure = st.number_input('Tenure (Months)', min_value=1, max_value=72, value=30)
    contract = st.selectbox('Contract (0: Month-to-month, 1: One year, 2: Two year)', [0, 1, 2])
    payment_method = st.selectbox('Payment Method (0: Electronic check, 1: Mailed check, 2: Bank transfer, 3: Credit card)', [0, 1, 2, 3])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.8, max_value=118.75, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=18.8, max_value=8684.8, value=1000.0)

    # Prédiction
    if st.button('Predict'):
        churn, churn_probability = predict_churn(gender, senior_citizen, tenure, contract, payment_method, 
                                                 monthly_charges, total_charges, threshold=threshold)
        
        st.write(f"📊 Probabilité de churn : {churn_probability:.2%}")  
        
        if churn == 1:
            st.write("🔴 Le client est susceptible de quitter (churn).")
        else:
            st.write("🟢 Le client est susceptible de rester.")

    # Afficher les coefficients du modèle
    if st.checkbox('Afficher les coefficients du modèle'):
        st.write(logreg_model.coef_)

elif menu == "Voir Dashboard":
    st.subheader("📊 Dashboard - Analyse du Churn")

    # Afficher le dashboard (image)
    st.image("dashboard.png", caption="Analyse du churn", use_container_width=True)

    # Bouton pour télécharger le PDF du dashboard
    with open("dashboard.pdf", "rb") as f:
        st.download_button("📥 Télécharger le Dashboard (PDF)", f, file_name="dashboard.pdf", mime="application/pdf")

elif menu == "Voir Rapport":
    st.subheader("📄 Rapport - Recommandations stratégiques")

    # Afficher un résumé des recommandations
    st.write("""
    **Principales recommandations pour réduire le churn :**  
    - 🔹 **Fidéliser les clients mensuels** : Offrir des réductions ou des avantages pour les encourager à passer à un abonnement annuel.  
    - 🔹 **Améliorer l’expérience des seniors** : Mettre en place une assistance prioritaire pour cette catégorie.  
    - 🔹 **Optimiser la facturation** : Encourager les paiements automatiques pour réduire le churn lié aux paiements manuels.  
    - 🔹 **Analyse des tarifs** : Vérifier si les clients ayant des factures élevées ont plus tendance à partir.  
    """)

    # Bouton pour télécharger le PDF du rapport
    with open("rapport.pdf", "rb") as f:
        st.download_button("📥 Télécharger le Rapport (PDF)", f, file_name="rapport.pdf", mime="application/pdf")
