import streamlit as st
import joblib
import numpy as np

# Charger le modèle de régression logistique
logreg_model = joblib.load('logreg_model3.pkl')

# Fonction de prédiction
def predict_churn(gender, senior_citizen, tenure, contract, payment_method, 
                  monthly_charges, total_charges, threshold=0.5):  # Default threshold = 0.3
    
    # Assurez-vous que l'ordre des caractéristiques corresponde à celui utilisé lors de l'entraînement
    input_data = np.array([float(gender), float(senior_citizen), float(tenure), float(contract),  
                       float(payment_method), float(monthly_charges), float(total_charges)]).reshape(1, -1)

    # Obtenir la probabilité de churn
    churn_probability = logreg_model.predict_proba(input_data)[:, 1][0]  # Probabilité de churn
    
    # Appliquer le seuil personnalisé
    churn_prediction = 1 if churn_probability >= threshold else 0
    
    return churn_prediction, churn_probability


# Interface utilisateur Streamlit
st.title('Customer Churn Prediction')

st.write("""
    Cette application prédit la probabilité qu'un client quitte l'entreprise (churn) en fonction de ses détails.
    Entrez les informations du client ci-dessous et obtenez la prédiction.
""")

# Option for the user to adjust the threshold
threshold = st.slider('Sélectionner le seuil de prédiction de churn', 0.0, 1.0, 0.5)

# Demander à l'utilisateur de saisir toutes les caractéristiques
gender = st.selectbox('Gender', ['Male', 'Female'])
gender = 1 if gender == 'Female' else 0

senior_citizen = st.selectbox('Senior Citizen (1: Yes, 0: No)', [0, 1])

tenure = st.number_input('Tenure (Months)', min_value=1, max_value=72, value=30)

contract = st.selectbox('Contract (0: Month-to-month, 1: One year, 2: Two year)', [0, 1, 2])

payment_method = st.selectbox('Payment Method (0: Electronic check, 1: Mailed check, 2: Bank transfer, 3: Credit card)', [0, 1, 2, 3])

monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.8, max_value=118.75, value=50.0)

total_charges = st.number_input('Total Charges ($)', min_value=18.8, max_value=8684.8, value=1000.0)

# Prédiction lorsque l'utilisateur clique sur le bouton
if st.button('Predict'):
    churn, churn_probability = predict_churn(gender, senior_citizen, tenure, contract, payment_method, 
                                             monthly_charges, total_charges, threshold=threshold)  # Apply threshold
    
    st.write(f"📊 Probabilité de churn : {churn_probability:.2%}")  # Show probability as percentage
    
    if churn == 1:
        st.write("🔴 Le client est susceptible de quitter (churn).")
    else:
        st.write("🟢 Le client est susceptible de rester.")

# Display model coefficients for transparency (optional)
if st.checkbox('Afficher les coefficients du modèle'):
    st.write(logreg_model.coef_)
