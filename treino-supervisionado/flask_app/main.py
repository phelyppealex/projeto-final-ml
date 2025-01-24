from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Carregar o modelo treinado
modelo = joblib.load("melhor_pipeline.pkl")

# Inicializar o app Flask
app = Flask(__name__)

# Página inicial
@app.route('/')
def index():
    return render_template('form.html')

# Rota para processar a previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Coletar os dados do formulário
        age_group = request.form['Age_Group']
        gender = request.form['Gender']
        smoking_status = request.form['Smoking_Status']
        physical_activity_level = request.form['Physical_Activity_Level']
        diet_quality = request.form['Diet_Quality']
        urban_rural = request.form['Urban_Rural']
        socioeconomic_status = request.form['Socioeconomic_Status']
        stress_level = request.form['Stress_Level']
        healthcare_access = request.form['Healthcare_Access']
        education_level = request.form['Education_Level']
        employment_status = request.form['Employment_Status']
        bmi = float(request.form['BMI'])
        alcohol_consumption = float(request.form['Alcohol_Consumption'])
        cholesterol_level = float(request.form['Cholesterol_Level'])
        air_pollution_index = float(request.form['Air_Pollution_Index'])
        region_heart_attack_rate = float(request.form['Region_Heart_Attack_Rate'])
        family_history = int(request.form['Family_History'])
        hypertension = int(request.form['Hypertension'])
        diabetes = int(request.form['Diabetes'])
        
        # Criando DataFrame
        d = {
            'Age_Group': [age_group],
            'Gender': [gender],
            'Smoking_Status': [smoking_status],
            'Physical_Activity_Level': [physical_activity_level],
            'Diet_Quality': [diet_quality],
            'Urban_Rural': [urban_rural],
            'Socioeconomic_Status': [socioeconomic_status],
            'Stress_Level': [stress_level],
            'Healthcare_Access': [healthcare_access],
            'Education_Level': [education_level],
            'Employment_Status': [employment_status],
            'Family_History': [family_history],
            'Hypertension': [hypertension],
            'Diabetes': [diabetes],
            'BMI': [bmi],
            'Alcohol_Consumption': [alcohol_consumption],
            'Cholesterol_Level': [cholesterol_level],
            'Air_Pollution_Index': [air_pollution_index],
            'Region_Heart_Attack_Rate': [region_heart_attack_rate],
        }
    
        data = pd.DataFrame(data=d)

        # Fazer a previsão
        previsao = modelo.predict(data)

        # Retornar o resultado
        resultado = "Você precisa cuidar da saúde: risco de ataque cardíaco." if previsao[0] == 1 else "É provável que a saúde do seu coração esteja boa."
        return render_template('response.html', result=resultado)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)