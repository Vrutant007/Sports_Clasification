# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from PIL import Image
import os
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Function to preprocess and classify the image
def classify_image(image):
    # Resize the image to 224x224 pixels as required by MobileNetV2
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict and decode the result
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1], decoded_predictions[0][0][2]  # Class label and confidence


# Custom CSS to style the app
st.markdown("""
    <style>
        /* Center the title and make it bold and colorful */
        h1, h2, h3 {
            text-align: center;
            font-weight: bold;
            color: #4B6A88;
        }
        /* Set the sidebar background color and text color */
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
            color: #4B6A88;
        }
        /* Style the main container */
        .main {
            background-color: #f0f2f6;
            padding: 1em;
            border-radius: 8px;
        }
        /* Add padding and border radius to buttons */
        .stButton>button {
            padding: 0.5em 1em;
            border-radius: 5px;
            color: #fff;
            background-color: #4B6A88;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to train the model
def train_model(X_train, y_train, task_type='regression'):
    if task_type == 'regression':
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Initialize model variable
trained_model = None

# Caching data load
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Load the data
data = load_data('sports.csv')

# Caching model training
@st.cache_resource
def train_model(X_train, y_train, task_type='regression'):
    if task_type == 'regression':
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", ("Overview", "Univariate Analysis", "Bivariate Analysis", "Correlation", "Prediction",'Image Viewing'))

# Main app title
st.title("Optimized EDA and Prediction App for Sports Classification")

# Overview Section
if section == "Overview":
    st.header("Dataset Overview")
    st.write("First few rows of the dataset:")
    st.write(data.head())
    st.write("**Dataset Shape**:", data.shape)
    st.write("**Data Types**:")
    st.write(data.dtypes)

# Univariate Analysis Section
elif section == "Univariate Analysis":
    st.header("Univariate Analysis")
    numeric_columns = data.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    analysis_type = st.selectbox("Choose the type of analysis:", ["Numerical", "Categorical"])
    
    if analysis_type == "Numerical" and numeric_columns:
        selected_num_col = st.selectbox("Choose a numerical column:", numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(data[selected_num_col], kde=True, ax=ax)
        st.pyplot(fig)
        
    elif analysis_type == "Categorical" and categorical_columns:
        selected_cat_col = st.selectbox("Choose a categorical column:", categorical_columns)
        
        # Limit to top 20 categories by count
        top_n = 20
        top_data = data[selected_cat_col].value_counts().head(top_n).reset_index()
        top_data.columns = [selected_cat_col, 'count']
        
        # Plot the top categories
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(y=selected_cat_col, x='count', data=top_data, ax=ax)
        ax.set_ylabel(selected_cat_col)
        ax.set_xlabel("Count")
        ax.set_title(f"Top 20 {selected_cat_col} by Count")
        st.pyplot(fig)
    else:
        st.write("No columns available for the selected analysis type.")

# Bivariate Analysis Section
elif section == "Bivariate Analysis":
    st.header("Bivariate Analysis")
    numeric_columns = data.select_dtypes(include=['float', 'int']).columns.tolist()
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("X-Axis", options=numeric_columns, index=0)
        y_axis = st.selectbox("Y-Axis", options=numeric_columns, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numerical columns for a bivariate analysis.")

# Correlation Section
elif section == "Correlation":
    st.header("Correlation Matrix")
    # Re-define numeric_columns for this section
    numeric_columns = data.select_dtypes(include=['float', 'int']).columns.tolist()
    if len(numeric_columns) > 1:
        # Only select numerical columns for the correlation matrix
        correlation_data = data[numeric_columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numerical columns to display a correlation matrix.")


# Prediction Section
elif section == "Prediction":
    st.header("Prediction Section")
    
    # Option to select a target column or upload an image
    prediction_type = st.radio("Choose prediction type", ["From Image Upload"])

    if prediction_type == "From Data":
        target_column = st.selectbox("Select the target column for prediction", options=data.columns)

        if target_column:
            # Drop the target column from the dataset to get the feature set
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Select only numerical columns for features
            numerical_X = X.select_dtypes(include=['float', 'int'])

            # Check if there are numerical columns remaining
            if numerical_X.shape[1] == 0:
                st.error("No numerical columns available for prediction after dropping the target column. Please select a different target.")
            else:
                # Preprocess data
                imputer = SimpleImputer(strategy="mean")
                scaler = StandardScaler()
                X_imputed = imputer.fit_transform(numerical_X)
                X_scaled = scaler.fit_transform(X_imputed)
                X = pd.DataFrame(X_scaled, columns=numerical_X.columns)

                # Train model
                with st.spinner("Training model..."):
                    model_type = 'regression' if y.dtype in ['float', 'int'] else 'classification'
                    trained_model = train_model(X, y, model_type)

                st.write("Model trained. Enter values to make a prediction.")

                # Create interactive inputs for prediction
                input_data = {col: st.number_input(f"Input {col}", value=0.0) for col in X.columns}

                if st.button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = trained_model.predict(input_df)[0] # Use trained_model here
                    st.write(f"**Predicted {target_column}:** {prediction}")

    elif prediction_type == "From Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Classify the image
        with st.spinner("Classifying the image..."):
            class_name, confidence = classify_image(image)
            st.write(f"**Predicted Category:** {class_name}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

# Image Viewing Section
elif section == "Image Viewing":
    st.header("Image Viewing Section")
    
    # Extract unique categories (assuming there's a column named 'category' in your CSV)
    image_columns = data['labels'].unique()  # Assuming 'train' is your category column, adjust as needed
    selected_sport = st.selectbox("Select a sport:", image_columns)

    # Filter dataset based on selected sport
    images_to_display = data[data['labels'] == selected_sport]['filepaths']  # Replace with the correct column

    # Display all images for the selected sport
    if not images_to_display.empty:
        for img_path in images_to_display:
            # Assuming your images are stored in a folder structure relative to your script
            full_img_path = os.path.join(os.getcwd(), img_path)  
            st.image(full_img_path, caption=selected_sport, use_column_width=True)
    else:
        st.write("No images found for the selected sport.")
