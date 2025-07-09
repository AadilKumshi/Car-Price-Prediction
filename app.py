import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #28a745;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
</style>
""", unsafe_allow_html=True)

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
            'data_cleaned': data_cleaned,
            'original_data': raw_data
        }
    
    except FileNotFoundError:
        st.error("⚠️ Dataset 'carproject.csv' not found. Please upload the dataset to continue.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Prediction App</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🔮 Price Prediction", "📈 Model Performance"]
    )
    
    # Load model and data
    model_data = load_and_train_model()
    
    if model_data is None:
        st.stop()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis(model_data)
    elif page == "🔮 Price Prediction":
        show_prediction_page(model_data)
    elif page == "📈 Model Performance":
        show_model_performance(model_data)

def show_home_page():
    """Display the home page"""
    st.markdown("""
    ## Welcome to the Car Price Prediction App! 🚗💰
    
    This application uses machine learning to predict used car prices based on various features such as:
    - **Brand** (BMW, Mercedes-Benz, Toyota, etc.)
    - **Body Type** (Sedan, Hatchback, SUV, etc.)
    - **Engine Volume** (in liters)
    - **Mileage** (kilometers driven)
    - **Engine Type** (Petrol, Diesel, Gas, etc.)
    - **Registration Status**
    
    ### How it works:
    1. **Data Analysis**: Explore the dataset and understand the relationships between features
    2. **Price Prediction**: Input car details to get an estimated price
    3. **Model Performance**: View detailed metrics about the prediction model's accuracy
    
    ### Model Details:
    - **Algorithm**: Linear Regression
    - **Data Preprocessing**: Outlier removal, log transformation, feature scaling
    - **Features**: 17 engineered features including categorical variables
    
    Navigate through the sidebar to explore different sections of the app!
    """)
    
    # Add some key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", "Linear Regression")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features Used", "17")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Split", "80/20")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Target Variable", "Log(Price)")
        st.markdown('</div>', unsafe_allow_html=True)

def show_data_analysis(model_data):
    """Display data analysis page"""
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    data = model_data['original_data']
    cleaned_data = model_data['data_cleaned']
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Features", len(data.columns))
    
    with col2:
        st.metric("Records after cleaning", len(cleaned_data))
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(data.head(), use_container_width=True)
    
    # Price distribution
    st.subheader("Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x='Price', nbins=50, title="Original Price Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Log price distribution
        log_prices = np.log(data['Price'])
        fig = px.histogram(x=log_prices, nbins=50, title="Log Price Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("Feature Relationships with Price")
    
    # Create subplot for multiple scatter plots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Mileage vs Price', 'Engine Volume vs Price', 'Year vs Price')
    )
    
    fig.add_trace(
        go.Scatter(x=data['Mileage'], y=data['Price'], mode='markers', 
                  name='Mileage', opacity=0.6),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data['EngineV'], y=data['Price'], mode='markers', 
                  name='Engine Volume', opacity=0.6),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=data['Year'], y=data['Price'], mode='markers', 
                  name='Year', opacity=0.6),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand analysis
    st.subheader("Price by Brand")
    brand_prices = data.groupby('Brand')['Price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=brand_prices.index, y=brand_prices['mean'], 
                    title="Average Price by Brand")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(values=brand_prices['count'], names=brand_prices.index, 
                    title="Car Count by Brand")
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model_data):
    """Display the prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Car Price Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter the car details below to get a price prediction:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", 
                               ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen'])
            body_type = st.selectbox("Body Type", 
                                   ['crossover', 'hatch', 'other', 'sedan', 'vagon', 'van'])
            engine_type = st.selectbox("Engine Type", 
                                     ['Diesel', 'Gas', 'Other', 'Petrol'])
            registration = st.selectbox("Registration", ['yes', 'no'])
        
        with col2:
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000)
            engine_volume = st.number_input("Engine Volume (L)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)
        
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
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🎯 Predicted Price</h3>
                <h1 style="color: #28a745; margin: 1rem 0;">${price_pred:,.2f}</h1>
                <p>Based on the provided car specifications</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature importance for this prediction
            st.subheader("Feature Impact Analysis")
            feature_impact = model_data['model'].coef_ * input_scaled[0]
            feature_names = model_data['feature_names']
            
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': feature_impact
            }).sort_values('Impact', key=abs, ascending=False)
            
            # Show top 10 most impactful features
            top_features = impact_df.head(10)
            
            fig = px.bar(top_features, x='Impact', y='Feature', orientation='h',
                        title="Top 10 Features Impacting This Prediction",
                        color='Impact', color_continuous_scale='RdYlBu')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

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
        'hatch': 8, 'other': 9, 'sedan': 10, 'vagon': 11, 'van': 12
    }
    if body_type in body_mapping:
        features[body_mapping[body_type]] = 1
    
    # Set engine type features (one-hot encoded, drop_first=True means 'Diesel' is the reference)
    engine_mapping = {
        'Gas': 13, 'Other': 14, 'Petrol': 15
    }
    if engine_type in engine_mapping:
        features[engine_mapping[engine_type]] = 1
    
    # Set registration feature (one-hot encoded, drop_first=True means 'no' is the reference)
    if registration == 'yes':
        features[16] = 1
    
    return features

def show_model_performance(model_data):
    """Display model performance metrics"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training R² Score", f"{model_data['train_score']:.4f}")
    
    with col2:
        st.metric("Testing R² Score", f"{model_data['test_score']:.4f}")
    
    with col3:
        overfitting = model_data['train_score'] - model_data['test_score']
        st.metric("Overfitting Gap", f"{overfitting:.4f}")
    
    # Feature importance
    st.subheader("Feature Importance (Model Coefficients)")
    
    coefficients = model_data['model'].coef_
    feature_names = model_data['feature_names']
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Plot feature importance
    fig = px.bar(coef_df.head(15), x='Coefficient', y='Feature', orientation='h',
                title="Feature Importance (Top 15 Features)",
                color='Coefficient', color_continuous_scale='RdYlBu')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show coefficient table
    st.subheader("All Model Coefficients")
    st.dataframe(coef_df, use_container_width=True)
    
    # Model interpretation
    st.subheader("Model Interpretation")
    st.write("""
    **Understanding the coefficients:**
    - **Positive coefficients** increase the predicted log price
    - **Negative coefficients** decrease the predicted log price
    - **Larger absolute values** have more impact on the prediction
    
    **Key insights:**
    - Luxury brands (BMW, Mercedes-Benz) typically have positive coefficients
    - Higher mileage typically reduces car value (negative coefficient)
    - Engine volume and certain body types can significantly impact price
    """)

if __name__ == "__main__":
    main()
