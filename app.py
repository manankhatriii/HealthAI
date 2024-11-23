import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tf_keras
import pandas as pd
import base64

col1, col2, col3= st.columns(3)
with col2:
    st.title("HealthAI")

image_path = "Models/maps.png"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
redirect_url = "https://www.google.com/maps/search/hospital+near+me"

st.sidebar.title("Select a Condition to Assess")
option = st.sidebar.selectbox("Choose a health condition:", ["Brain Tumor", "Diabetes", 
                                                             "Chronic Liver infection", "Lung Cancer", 
                                                             "Metal Toxicity", "Tuberculosis"]) 

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")      

st.sidebar.write(f":green[Nearby hospitals: ]")
st.sidebar.markdown(
                f'<a href="{redirect_url}" target="_blank">'
                f'<img src="data:image/png;base64,{encoded_image}" alt="Map" style="width:100%; max-width:450px; cursor:pointer; border:none;"></a>',
                unsafe_allow_html=True
                )

if option == "Brain Tumor":
    with st.spinner("Loading..."):
        brain_tumor_model = tf_keras.models.load_model("Models/braintumor_model.h5")
    st.header("Brain Tumor Risk Assessment")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Analyzing..."):
        
            col1, col2, col3= st.columns(3)
            with col2:
                st.image(image)
            
            st.write("-"*40)
            image = image.resize((299, 299))
            image_array = np.array(image) / 255.0
            prediction = brain_tumor_model.predict(np.expand_dims(image_array, axis=0))
            classes = ['glioma', 'healthy', 'pituitary']
            predicted_class = classes[np.argmax(prediction)]
            
            if predicted_class == "glioma" and np.max(prediction) > 0.7:
                st.write(f":red[Early stage of GLIOMA.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Diffuse midline gliomas")
                    st.write("- Mass spread in the middle area of the brain")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Clinical nanograms")
                    st.write("- Liquid biopsies")
                
            elif predicted_class == "pituitary" and np.max(prediction) > 0.7:
                st.write(f":red[Early stage of PITUITARY.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1: 
                    st.write("Key insights: ")
                    st.write("- Enlargement of the pituitary gland")
                    st.write("- Abnormal shape of the pituitary gland")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Drug therapy")
                    st.write("- Radiation therapy")
                    
            else:
                st.write(f":green[You are NOT prone to brain tumor! Keep up the healthy lifestyle!]")
                st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                
        st.success("Prediction completed!")

elif option == "Diabetes":
    with st.spinner("Loading..."):
        diabetes_model = joblib.load("Models/diabetes_model.joblib")
    st.header("Diabetes Risk Assessment")

    with st.form("diabetes_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=0)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col3:
            polyuria = st.selectbox("Polyuria", ["Yes", "No"])
        with col4:
            polydipsia = st.selectbox("Polydipsia", ["Yes", "No"])
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
        with col6:
            weakness = st.selectbox("Weakness", ["Yes", "No"])
        with col7:
            polyphagia = st.selectbox("Polyphagia", ["Yes", "No"])
        with col8:
            genital_thrush = st.selectbox("Genital Thrush", ["Yes", "No"])
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
        with col10:
            itching = st.selectbox("Itching", ["Yes", "No"])
        with col11:
            irritability = st.selectbox("Irritability", ["Yes", "No"])
        with col12:
            delayed_healing = st.selectbox("Delayed Healing", ["Yes", "No"])
        col13, col14, col15, col16 = st.columns(4)
        with col13:
            partial_paresis = st.selectbox("Partial Paresis", ["Yes", "No"])
        with col14:
            muscle_stiffness = st.selectbox("Muscle Stiffness", ["Yes", "No"])
        with col15:
            alopecia = st.selectbox("Alopecia", ["Yes", "No"])
        with col16:
            obesity = st.selectbox("Obesity", ["Yes", "No"])
        
        with st.expander("Help:"):
            st.write("- Age: The age of the individual in years.") 
            st.write("- Gender: The gender of the individual (Male or Female).")
            st.write("- Polyuria: Indicates whether the individual experiences excessive urination (Yes or No).")
            st.write("- Polydipsia: Indicates whether the individual experiences excessive thirst (Yes or No).")
            st.write("- Sudden weight loss: Indicates whether the individual has experienced sudden weight loss (Yes or No).")
            st.write("- Weakness: Indicates whether the individual feels weakness (Yes or No).")
            st.write("- Polyphagia: Indicates whether the individual experiences excessive hunger (Yes or No).")
            st.write("- Genital thrush: Indicates whether the individual has experienced genital thrush (Yes or No).")
            st.write("- Visual blurring: Indicates whether the individual has experienced blurred vision (Yes or No).")
            st.write("- Itching: Indicates whether the individual experiences frequent itching (Yes or No).")
            st.write("- Irritability: Indicates whether the individual experiences irritability (Yes or No).")
            st.write("- Delayed healing: Indicates whether the individual experiences delayed healing of wounds (Yes or No).")
            st.write("- Partial paresis: Indicates whether the individual has partial paralysis (Yes or No).")
            st.write("- Muscle stiffness: Indicates whether the individual experiences muscle stiffness (Yes or No).")
            st.write("- Alopecia: Indicates whether the individual has experienced hair loss (Yes or No).")
        
        col1, col2, col3, col4, col5= st.columns(5)
        with col3:
            submit_button = st.form_submit_button("Predict")
        
    if submit_button:
        with st.spinner("Analyzing..."):
            input_data = np.array([[age, 1 if gender == "Male" else 0,
                                    1 if polyuria == "Yes" else 0,
                                    1 if polydipsia == "Yes" else 0,
                                    1 if sudden_weight_loss == "Yes" else 0,
                                    1 if weakness == "Yes" else 0,
                                    1 if polyphagia == "Yes" else 0,
                                    1 if genital_thrush == "Yes" else 0,
                                    1 if visual_blurring == "Yes" else 0,
                                    1 if itching == "Yes" else 0,
                                    1 if irritability == "Yes" else 0,
                                    1 if delayed_healing == "Yes" else 0,
                                    1 if partial_paresis == "Yes" else 0,
                                    1 if muscle_stiffness == "Yes" else 0,
                                    1 if alopecia == "Yes" else 0,
                                    1 if obesity == "Yes" else 0]])
        
            prediction = diabetes_model.predict(input_data)
            if prediction[0] == 0:
                st.write(f":green[You are NOT PRONE to having Diabetes. Keep up the healthy lifestyle!]")
            else:
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write(f":red[Early stage of DIABETES.]")
                    
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Feeling more thirsty")
                    st.write("- Weight loss")
                    st.write("- Feeling tired and weak")
                    st.write("- Feeling irritable")
                    st.write("- Having slow healing sores")
                    
                with col2:    
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Cut sugar and refined carbs")
                    st.write("- Quit smoking (if applicable)")
                    st.write("- Regular exercise")
                    st.write("- Drink plenty of water")
                    st.write("- Maintain a healthy weight")
                    st.write("- Manage stress")
                image = Image.open("Models/diabetes_correlation.png")
                st.image(image)
                
        st.success("Prediction completed!")
    

elif option == "Chronic Liver infection":
    with st.spinner("Loading..."):
        hepatitis_model = tf_keras.models.load_model("Models/hepatitisc_model.h5")
    st.header("Chronic Liver infection Risk Assessment")
    
    with st.form("hepatitis_form"):
        col1, col2, col3= st.columns(3)
        with col1: 
            age= st.number_input("Age", min_value= 1)
        with col2:
            gender= st.selectbox("Gender", ["Male", "Female"])
        with col3:
            albumin= st.number_input("Albumin")
        
        col4, col5, col6= st.columns(3)
        with col4:
            alkaline_phosphatase = st.number_input("Alkaline Phosphatase")
        with col5:
            alanine_aminotransferase = st.number_input("Alanine Aminotransferase")
        with col6:
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
        
        col7, col8, col9= st.columns(3)
        with col7:
            bilirubin = st.number_input("Bilirubin")
        with col8:
            cholesterol = st.number_input("Cholesterol")
        with col9:
            creatinine = st.number_input("Creatinine")
        col10, col11= st.columns(2)
        with col10:
            gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase")
        with col11:        
            protein = st.number_input("Protein")
                
        with st.expander("Help"):
            data_units = {
                "Column": [
                    "Age", "Gender", "Albumin", "Alkaline_Phosphatase",
                    "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Bilirubin",
                    "Cholesterol", "Creatinine", "Gamma_Glutamyl_Transferase", "Protein"
                ],
                "Units": [
                    "Years", "-", "g/dL", "IU/L", "IU/L", "IU/L", "mg/dL", "mg/dL", 
                    "mg/dL", "IU/L", "g/dL"
                ],
                "Description": [
                    "Patient's age in years",
                    "Male / Female",
                    "Serum albumin concentration",
                    "Alkaline phosphatase enzyme level",
                    "ALT enzyme level, indicative of liver health",
                    "AST enzyme level, indicative of liver health",
                    "Bilirubin concentration in the blood",
                    "Total cholesterol level",
                    "Serum creatinine, indicative of kidney function",
                    "GGT enzyme level, associated with liver/bile ducts",
                    "Total protein concentration in the blood"
                ]
            }
            df_units = pd.DataFrame(data_units)
            st.dataframe(df_units, width=700, height=425)
            
        col1, col2, col3, col4, col5= st.columns(5)
        with col3:
            submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        with st.spinner("Analyzing..."):
            input_data = np.array([[age, 
                                    1 if gender == "Male" else 0, 
                                    albumin, 
                                    alkaline_phosphatase,
                                    alanine_aminotransferase, 
                                    aspartate_aminotransferase,
                                    bilirubin, 
                                    cholesterol, 
                                    creatinine,
                                    gamma_glutamyl_transferase, 
                                    protein]])
            
            prediction = hepatitis_model.predict(input_data)
            classes= ["healthy", "hepatitis", "fibrosis", "cirrhosis"]
            predicted_class= classes[np.argmax(prediction)]
            
            if predicted_class == "healthy":
                st.write(f":green[You are NOT prone to Liver infection! Keep up the healthy lifestyle!]")
                st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
            
            elif predicted_class == "hepatitis":
                st.write(f":red[Early stage of HEPATITIS.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Dark Urine")
                    st.write("- Pale stool")
                    st.write("- Pain in right upper abdomen")
                    st.write("- Weight loss")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Antiviral medications")
                    st.write("- Pan-genotypic therapy")
                    st.write("- Exercise regularly")
                    
            elif predicted_class == "cirrhosis" or predicted_class == "fibrosis":
                st.write(f":red[Early stage of CIRRHOSIS.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Nausea")
                    st.write("- Visible blood vessels")
                    st.write("- Pain in upper abdomen")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Avoid alchohol")
                    st.write("- Exercise regularly")
                
            image = Image.open("Models/liver_correlation.png")
            st.image(image)
        st.success("Prediction completed!")
               
elif option == "Lung Cancer":
    with st.spinner("Loading..."):
        lung_cancer_model = tf_keras.models.load_model("Models/lungcancer_model.h5")
    st.header("Lung Cancer Risk Assessment")
    uploaded_file = st.file_uploader("Upload a Lung CT Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner("Analyzing..."):
        
            col1, col2, col3= st.columns(3)
            with col2:
                st.image(image, caption="Uploaded Lung CT Image")
            
            st.write("-"*40)
            image= image.convert("RGB")
            image = image.resize((299, 299))
            image_array = np.array(image) / 255.0
            prediction = lung_cancer_model.predict(np.expand_dims(image_array, axis=0))
            classes= ["adenocarcinoma", "large cell carcinoma", "normal", "squamous cell carcinoma"]
            predicted_class= classes[np.argmax(prediction)]
            
            if predicted_class == "normal":
                st.write(f":green[You are NOT prone to lung cancer! Keep up the healthy lifestyle!]")
                st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
            
            elif predicted_class == "adenocarcinoma":
                st.write(f":red[Early stage of ADENOCARCINOMA.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- A 'halo sign' around a mass, indicating inflammation or edema")
                    st.write("- Abnormal lymph nodes or mediastinal involvement")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Surgery")
                    st.write("- Chemotherapy")
            
            elif predicted_class == "large cell carcinoma":
                st.write(f":red[Early stage of LARGE CELL CARCINOMA.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Large mass in the lung, with irregular borders")
                    st.write("- Enlargement of lymph nodes")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Surgery")
                    st.write("- Chemotherapy")
                
            elif predicted_class == "squamous cell carcinoma":
                st.write(f":red[Early stage of SQUAMOUS CELL CARCINOMA.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- Solitary nodule or mass in the lung")
                    st.write("- Increased size or number of lymph nodes")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- Surgery")
                    st.write("- Chemotherapy")
                
        st.success("Prediction completed!")

elif option == "Metal Toxicity":
    with st.spinner("Loading..."):
        metal_toxicity_fit = joblib.load("Models/metaltoxicity_fit.joblib")
        metal_toxicity_arsenic = joblib.load("Models/metaltoxicity_arsenic.joblib")
        metal_toxicity_mercury = joblib.load("Models/metaltoxicity_mercury.joblib")
        metal_toxicity_nickel = joblib.load("Models/metaltoxicity_nickel.joblib")
    st.header("Metal Toxicity Risk Assessment")
    
    with st.form("metal_form"):
        col1, col2, col3= st.columns(3)
        with col1:
            subsample_weights = st.number_input("Subsample Weights")
        with col2:
            arsenous_acid = st.number_input("Arsenous Acid")
        with col3:
            arsenic_acid = st.number_input("Arsenic Acid")
        col4, col5, col6= st.columns(3)
        with col4:
            arsenobetaine = st.number_input("Arsenobetaine")
        with col5:    
            arsenocholine = st.number_input("Arsenocholine")
        with col6:
            dimethylarsinic_acid = st.number_input("Dimethylarsinic Acid")
        col7, col8, col9= st.columns(3)           
        with col7:
            monomethylarsonic_acid = st.number_input("Monomethylarsonic Acid")
        with col8:    
            mercury_level = st.number_input("Mercury Level")
        with col9:
            nickel_level = st.number_input("Nickel Level")
            
        with st.expander("Help"):
            metal_toxicity_data_units = {
                "Column": [
                    "Subsample weights", "Arsenous acid", "Arsenic acid", "Arsenobetaine", 
                    "Arsenocholine", "Dimethylarsinic Acid", "Monomethylarsonic Acid", 
                    "Mercury level", "Nickel level"
                ],
                "Units": [
                    "µg/L", "µg/L", "µg/L", "µg/L", "µg/L", "µg/L", "µg/L", "µg/L", "µg/L"
                ],
                "Description": [
                    "Total sample weight in micrograms per liter (µg/L)",
                    "Concentration of arsenous acid (As(III)) in the sample",
                    "Concentration of arsenic acid (As(V)) in the sample",
                    "Concentration of arsenobetaine, a common arsenic compound",
                    "Concentration of arsenocholine, an organic arsenic compound",
                    "Concentration of dimethylarsinic acid (DMA), a metabolite of arsenic",
                    "Concentration of monomethylarsonic acid (MMA), a metabolite of arsenic",
                    "Mercury concentration level in the sample",
                    "Nickel concentration level in the sample"
                ]
            }
            df_metal_toxicity_units = pd.DataFrame(metal_toxicity_data_units)
            st.dataframe(df_metal_toxicity_units, width=800, height=352)
        col1, col2, col3, col4, col5= st.columns(5)
        with col3:
            submit_button = st.form_submit_button("Predict")
        
    if submit_button:
        with st.spinner("Analyzing..."):
            input_data = np.array([[subsample_weights, arsenous_acid, arsenic_acid,
                                    arsenobetaine, arsenocholine, dimethylarsinic_acid,
                                    monomethylarsonic_acid, mercury_level, nickel_level]])
            
            fit_prediction= metal_toxicity_fit.predict(input_data)
            
            if int(np.max(fit_prediction)) < 0.45:
                st.write(f":green[You are NOT PRONE to Metal Poisoning. Keep up the healthy lifestyle!]")
            else:
                arsenic_prediction= metal_toxicity_arsenic.predict(input_data)
                mercury_prediction= metal_toxicity_mercury.predict(input_data)
                nickel_prediction= metal_toxicity_nickel.predict(input_data)
                
                if np.max(arsenic_prediction) > 0.75:
                    st.write(f":red[Early stage of ARSENICOSIS.]")
                    col1, col2, col3= st.columns(3)
                    with col2:
                        st.write("Prediction confidence: ", np.round(np.max(arsenic_prediction), 2) * 100, " %")
                    col1, col2= st.columns(2)
                    with col1:
                        st.write("Key insights: ")
                        st.write("- Skin pigmentation and lesions")
                        st.write("- Hard patches on palms and soles")
                        st.write("- Stomach pain")
                    with col2:
                        st.write("Preventive measures: ")
                        st.write("- Please consult a doctor.")
                        st.write("- Chelation therapy")
                        st.write("- Remove arsenic content from body using administerating agents")
                        
                if np.max(mercury_prediction) > 0.75:
                    st.write(f":red[Early stage of MERCURY POISONING.]")
                    col1, col2, col3= st.columns(3)
                    with col2:
                        st.write("Prediction confidence: ", np.round(np.max(mercury_prediction), 2) * 100, " %")
                    col1, col2= st.columns(2)
                    with col1:
                        st.write("Key insights: ")
                        st.write("- Metallic taste in your mouth")
                        st.write("- Nausea or vomiting")
                        st.write("- Bleeding or swollen gums")
                    with col2:
                        st.write("Preventive measures: ")
                        st.write("- Please consult a doctor.")
                        st.write("- Fastric lavage")
                        st.write("- Hydration")
                        
                if np.max(nickel_prediction) > 0.75:
                    st.write(f":red[Early stage of NICKEL ALLERGIC CONTACT DERMATITIS.]")
                    col1, col2, col3= st.columns(3)
                    with col2:
                        st.write("Prediction confidence: ", np.round(np.max(nickel_prediction), 2) * 100, " %")
                    col1, col2= st.columns(2)
                    with col1:
                        st.write("Key insights: ")
                        st.write("- Rash or bumps on the skin")
                        st.write("- Itching, which may be severe")
                        st.write("- Dry patches of skin that may resemble a burn")
                    with col2:
                        st.write("Preventive measures: ")
                        st.write("- Please consult a doctor.")
                        st.write("- Topical corticosteroid creams or ointments")
                        st.write("- Avoiding further exposure to nickel")
            image= Image.open("Models/metal_correlation.png")
            st.image(image)
                
        st.success("Prediction completed!")

elif option == "Tuberculosis":
    with st.spinner("Loading..."):
        tb_model = tf_keras.models.load_model("Models/tuberculosis_model.h5")
    st.header("Tuberculosis Risk Assessment")
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner("Analyzing..."):
            col1, col2, col3= st.columns(3)
            with col2:
                st.image(image, caption="Uploaded Chest X-ray Image")

            st.write("-"*40)
            image= image.convert("RGB")
            image = image.resize((299, 299))
            image_array = np.array(image) / 255.0
            prediction = tb_model.predict(np.expand_dims(image_array, axis=0))
            
            if prediction[0] < 0.4:
                st.write(f":green[You are NOT PRONE to Tuberculosis. Keep up the healthy lifestyle!]")
            else:
                st.write(f":red[Early stage of TUBERCULOSIS.]")
                col1, col2, col3= st.columns(3)
                with col2:
                    st.write("Prediction confidence: ", round(np.max(prediction) * 100, 2), " %")
                col1, col2= st.columns(2)
                with col1:
                    st.write("Key insights: ")
                    st.write("- 1-2 cm diameter nodules or masses in the upper lobes of the lungs")
                    st.write("- Bilateral hilar lymphadenopathy (enlarged lymph nodes in the chest)")
                    st.write("- Intrapulmonary cavitation")
                with col2:
                    st.write("Preventive measures: ")
                    st.write("- Please consult a doctor.")
                    st.write("- 4-drug regimen(INH, RIF, PZA, EMB)")
                    st.write("- Avoiding further exposure to nickel")
        st.success("Prediction completed!")
