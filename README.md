# Car Price Prediction Streamlit App 🚗💰

A comprehensive web application for predicting used car prices using machine learning. Built with Streamlit, this app provides an interactive interface for data exploration, price prediction, and model performance analysis.

## Features

- **🏠 Home Page**: Overview of the application and model details
- **📊 Data Analysis**: Interactive data exploration with visualizations
- **🔮 Price Prediction**: Real-time car price prediction based on user inputs
- **📈 Model Performance**: Detailed analysis of model metrics and feature importance

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data (if you don't have carproject.csv):**
   ```bash
   python generate_sample_data.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Dataset

The app expects a CSV file named `carproject.csv` with the following columns:
- `Brand`: Car manufacturer (Audi, BMW, Mercedes-Benz, etc.)
- `Model`: Car model name
- `Year`: Manufacturing year
- `Body`: Body type (sedan, hatch, crossover, etc.)
- `Mileage`: Kilometers driven
- `EngineV`: Engine volume in liters
- `Engine Type`: Fuel type (Petrol, Diesel, Gas, Other)
- `Registration`: Registration status (yes/no)
- `Price`: Car price (target variable)

If you don't have this dataset, run `generate_sample_data.py` to create a synthetic dataset.

## Model Details

- **Algorithm**: Linear Regression
- **Preprocessing**: 
  - Outlier removal (99th percentile for Price and Mileage)
  - Log transformation of target variable
  - Feature scaling using StandardScaler
  - One-hot encoding for categorical variables
- **Features**: 17 engineered features including brand, body type, engine specifications
- **Validation**: 80/20 train-test split

## Usage

### Home Page
- Overview of the application
- Model specifications and key metrics

### Data Analysis
- Dataset overview and statistics
- Price distribution analysis
- Feature relationship exploration
- Brand analysis with interactive charts

### Price Prediction
- Input form for car specifications
- Real-time price prediction
- Feature impact analysis for individual predictions

### Model Performance
- Training and testing R² scores
- Feature importance visualization
- Model coefficient analysis
- Interpretation guidelines

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **Styling**: Custom CSS for enhanced UI

## File Structure

```
streamlit-app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── generate_sample_data.py   # Script to create sample dataset
├── carproject.csv           # Dataset (generated or provided)
└── README.md               # This file
```

## Features Explained

### Data Preprocessing
1. **Missing Value Handling**: Removes rows with missing values
2. **Outlier Removal**: Filters extreme values in Price, Mileage, EngineV, and Year
3. **Log Transformation**: Applies natural log to price for better model performance
4. **Feature Engineering**: Creates dummy variables for categorical features
5. **Multicollinearity**: Removes Year variable due to high correlation

### Model Features
The model uses 17 features:
- **Continuous**: Mileage, Engine Volume
- **Brand Dummies**: BMW, Mercedes-Benz, Mitsubishi, Renault, Toyota, Volkswagen
- **Body Type Dummies**: Hatch, Other, Sedan, Vagon, Van
- **Engine Type Dummies**: Gas, Other, Petrol
- **Registration**: Yes/No

### Prediction Process
1. User inputs car specifications through the web form
2. Features are encoded and scaled using the trained scaler
3. Model predicts log price
4. Result is transformed back to actual price
5. Feature impact analysis shows which features influenced the prediction

## Customization

You can customize the app by:
- Modifying the CSS styling in the `st.markdown()` sections
- Adding new visualizations in the data analysis section
- Implementing different machine learning models
- Adding new features to the prediction form
- Changing the color schemes and layout

## Performance Metrics

The app displays various performance metrics:
- **R² Score**: Coefficient of determination
- **Feature Importance**: Model coefficients sorted by absolute value
- **Overfitting Analysis**: Difference between training and testing scores

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`
2. **Dataset Not Found**: Run `python generate_sample_data.py` to create sample data
3. **Memory Issues**: Reduce the dataset size in `generate_sample_data.py`
4. **Performance Issues**: Consider using `@st.cache_data` for heavy computations

## Future Enhancements

- Add more machine learning algorithms (Random Forest, XGBoost)
- Implement hyperparameter tuning
- Add cross-validation metrics
- Include confidence intervals for predictions
- Add data upload functionality
- Implement model comparison features

## Contributing

Feel free to fork this repository and submit pull requests for any improvements!

## License

This project is open source and available under the MIT License.
