# HealthAI: Early Disease Prediction App

## Overview
**HealthAI** is a powerful Streamlit-based web application designed for early detection and prediction of multiple diseases. The app integrates machine learning and deep learning models to analyze various types of data and provide health insights that could help users take proactive measures for their well-being.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Data](#models-and-data)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)


## Features
- **Disease Predictions**: Supports predictions for:
  - Brain Tumor (classification into glioma, meningioma, pituitary, or healthy)
  - Diabetes (prone or not prone)
  - Hepatitis C (prone to hepatitis, fibrosis, cirrhosis, or healthy)
  - Kidney Dysfunction (detection based on lab values)
  - Lung Cancer (classification into adenocarcinoma, large cell carcinoma, squamous cell carcinoma, or healthy)
  - Metal Toxicity (arsenicosis, mercury poisoning, nickel dermatitis, or healthy)
  - Tuberculosis (detection from X-ray images)
- **Interactive User Input**: Users can input medical data or upload images for predictions.
- **Detailed Reports**: Generates downloadable PDF reports with input values, prediction results, key insights, and preventive measures.
- **Visual Insights**: Displays correlation plots for specific conditions to show feature importance.

## Project Structure
.
├── Models
│   ├── braintumor_model.h5
│   ├── diabetes_correlation.png
│   ├── diabetes_model.joblib
│   ├── hepatitisc_model.h5
│   ├── liver_correlation.png
│   ├── lungcancer_model.h5
│   ├── maps.png
│   ├── metal_correlation.png
│   ├── metaltoxicity_arsenic.joblib
│   ├── metaltoxicity_fit.joblib
│   ├── metaltoxicity_mercury.joblib
│   ├── metaltoxicity_nickel.joblib
│   └── tuberculosis_model.h5
├── README.md
├── TestingData
│   ├── BrainTumor
│   │   ├── Glioma (1).jpg
│   │   ├── Glioma (2).jpg
│   │   ├── Glioma (3).jpg
│   │   ├── Healthy (1).jpg
│   │   ├── Healthy (2).jpg
│   │   ├── Healthy (3).jpg
│   │   ├── Meningioma (1).jpg
│   │   ├── Meningioma (2).jpg
│   │   ├── Meningioma (3).jpg
│   │   ├── Pituitary (1).jpg
│   │   ├── Pituitary (2).jpg
│   │   └── Pituitary (3).jpg
│   ├── ChronicLiverInfection
│   │   └── ChronicLiverInfection.csv
│   ├── Diabetes
│   │   └── Diabetes.csv
│   ├── LungCancer
│   │   ├── Adenocarcinoma (1).png
│   │   ├── Adenocarcinoma (2).png
│   │   ├── Adenocarcinoma (3).png
│   │   ├── LargeCellCarcinoma (1).png
│   │   ├── LargeCellCarcinoma (2).png
│   │   ├── LargeCellCarcinoma (3).png
│   │   ├── Normal (1).png
│   │   ├── Normal (2).png
│   │   ├── Normal (3).png
│   │   ├── SquamousCellCarcinoma (1).png
│   │   ├── SquamousCellCarcinoma (2).png
│   │   └── SquamousCellCarcinoma (3).png
│   ├── MetalToxicity
│   │   └── Metal.csv
│   └── Tuberculosis
│       ├── Healthy (1).png
│       ├── Healthy (2).png
│       ├── Healthy (3).png
│       ├── TB (1).png
│       ├── TB (2).png
│       └── TB (3).png
├── app.py
└── requirements.txt


## Technologies Used
- **Python**: Core language for development.
- **Streamlit**: For building the web interface.
- **TensorFlow/Keras**: Deep learning model development.
- **Scikit-learn**: Machine learning algorithms and model evaluation.
- **Pandas**: Data manipulation and analysis.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/HealthAI.git
   cd HealthAI
2. **Requirements** (joblib==1.4.2
                      numpy==2.1.3
                      pandas==2.2.3
                      Pillow==11.0.0
                      streamlit==1.37.1
                       tf_keras==2.17.0)
   pip install -r requirements.txt


## Usage
   1.  **Run the app**
   streamlit run app.py
   2. **Interact with app**
      - Choose a disease from the dropdown menu.
      - Enter the required input values or upload an image for analysis.

## Models and Data
**Pre-trained Models**: Stored in the Models directory.

**Testing Data**: Sample data for each disease is provided in the TestingData directory.

## Future Enhancements
- User Authentication: To allow users to save their prediction history.
- Real-time Data Updates: Integrate with health databases for up-to-date predictions.
- Multi-language Support: Make the app accessible to non-English speakers.
- Mobile Optimization: Enhance the UI/UX for mobile users.


## Contributing
This project is intended for **submission purposes only**. Permissions for updates, modifications, or usage of the code are **prohibited** until **June 2025**. If you have questions or require assistance regarding the project, please contact the project owner directly.
