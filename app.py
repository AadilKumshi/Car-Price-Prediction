import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="",
    layout="centered"
)

# Cache the model training function
@st.cache_data
def load_and_train_model():
    """Load data and train the model"""
    try:
        # Load the data
        raw_data = pd.read_csv('carproject.csv')
        
        # Data preprocessing
        data = raw_data.drop(['Model'], axis=1)
        data = data.dropna()
        
        # Remove outliers
        q = data['Price'].quantile(0.99)
        data = data[data['Price'] < q]
        
        q1 = data['Mileage'].quantile(0.99)
        data = data[data['Mileage'] < q1]
        
        data = data[data['EngineV'] < 6.5]
        
        q3 = data['Year'].quantile(0.01)
        data = data[data['Year'] > q3]
        
        data_cleaned = data.reset_index(drop=True)
        
        # Transform target variable
        log_price = np.log(data_cleaned['Price'])
        data_cleaned['log_price'] = log_price
        data_cleaned = data_cleaned.drop(['Price'], axis=1)
        
        # Remove multicollinearity
        data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
        
        # Create dummy variables
        data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
        
        # Select final columns
        cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
               'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
               'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
               'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
               'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
        
        data_preprocessed = data_with_dummies[cols]
        
        # Prepare features and target
        targets = data_preprocessed['log_price']
        inputs = data_preprocessed.drop(['log_price'], axis=1)
        
        # Scale the data
        scaler = StandardScaler()
        inputs_scaled = scaler.fit_transform(inputs)
        
        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            inputs_scaled, targets, test_size=0.2, random_state=365
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(x_train, y_train)
        
        # Calculate metrics
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': inputs.columns.tolist(),
            'train_score': train_score,
            'test_score': test_score,
            'original_data': raw_data
        }
    
    except FileNotFoundError:
        st.error("⚠️ Dataset 'carproject.csv' not found. Please upload the dataset to continue.")
        return None

def prepare_input_features(brand, body_type, engine_type, registration, mileage, engine_volume):
    """Prepare input features for prediction"""
    # Initialize all features to 0
    features = np.zeros(17)  # Total number of features
    
    # Set continuous features
    features[0] = mileage  # Mileage
    features[1] = engine_volume  # EngineV
    
    # Set brand features (one-hot encoded, drop_first=True means 'Audi' is the reference)
    brand_mapping = {
        'BMW': 2, 'Mercedes-Benz': 3, 'Mitsubishi': 4, 
        'Renault': 5, 'Toyota': 6, 'Volkswagen': 7
    }
    if brand in brand_mapping:
        features[brand_mapping[brand]] = 1
    
    # Set body type features (one-hot encoded, drop_first=True means 'crossover' is the reference)
    body_mapping = {
        'Hatch': 8, 'Other': 9, 'Sedan': 10, 'Vagon': 11, 'Van': 12
    }
    if body_type.lower() in body_mapping:
        features[body_mapping[body_type.lower()]] = 1
    
    # Set engine type features (one-hot encoded, drop_first=True means 'Diesel' is the reference)
    engine_mapping = {
        'Gas': 13, 'Other': 14, 'Petrol': 15
    }
    if engine_type in engine_mapping:
        features[engine_mapping[engine_type]] = 1
    
    # Set registration feature (one-hot encoded, drop_first=True means 'no' is the reference)
    if registration.lower() == 'yes':
        features[16] = 1
    
    return features

def main():
    # Title
    st.title("Car Price Prediction")
    st.write("Get an estimated price for your used car using machine learning!")
    
    # Load model and data
    model_data = load_and_train_model()
    
    if model_data is None:
        st.stop()
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["Predict Price", "Data Overview", "Model Info"])
    
    with tab1:
        st.header("Predict Your Car Price")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                brand = st.selectbox("Brand", 
                                   ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen'])
                body_type = st.selectbox("Body Type", 
                                       ['Crossover', 'Hatch', 'Sedan', 'Vagon', 'Van', 'Other'])
                engine_type = st.selectbox("Engine Type", 
                                         ['Diesel', 'Gas', 'Petrol', 'Other'])
            
            with col2:
                registration = st.selectbox("Registration", ['Yes', 'No'])
                mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=65, step=1000)
                engine_volume = st.number_input("Engine Volume (L)", min_value=0.5, max_value=6.0, value=2.5, step=0.1)
            
            submitted = st.form_submit_button("🎯 Predict Price", use_container_width=True)
            
            if submitted:
                # Prepare input features
                input_features = prepare_input_features(
                    brand, body_type, engine_type, registration, mileage, engine_volume
                )
                
                # Scale the features
                input_scaled = model_data['scaler'].transform([input_features])
                
                # Make prediction
                log_price_pred = model_data['model'].predict(input_scaled)[0]
                price_pred = np.exp(log_price_pred)
                
                # Display result
                st.success(f"🎯 **Predicted Price: ${price_pred:,.2f}**")
                
                # Show input summary
                st.info(f"""
                **Your car details:**
                - Brand: {brand}
                - Body Type: {body_type}
                - Engine Type: {engine_type}
                - Registration: {registration}
                - Mileage: {mileage:,} km
                - Engine Volume: {engine_volume} L
                """)
    
    with tab2:
        st.header("Dataset Overview")
        
        data = model_data['original_data']
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cars", len(data))
        with col2:
            st.metric("Average Price", f"${data['Price'].mean():,.0f}")

        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Simple charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig = px.histogram(data, x='Price', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Price by Brand")
            brand_avg = data.groupby('Brand')['Price'].mean().sort_values(ascending=False)
            fig = px.bar(x=brand_avg.index, y=brand_avg.values)
            fig.update_xaxes(title="Brand")
            fig.update_yaxes(title="Average Price ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Performance")
        
        # Model metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training Accuracy", f"{model_data['train_score']:.3f}")
        
        with col2:
            st.metric("Testing Accuracy", f"{model_data['test_score']:.3f}")
        
        # Model info
        st.subheader("Model Details")
        st.write("""
        - **Algorithm**: Linear Regression
        - **Features**: 17 engineered features
        - **Data Processing**: Outlier removal, log transformation, feature scaling
        - **Validation**: 80/20 train-test split
        """)

if __name__ == "__main__":
    main()
