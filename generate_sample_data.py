import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define the parameters for generating synthetic data
n_samples = 1000

# Generate synthetic car data
brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Toyota', 'Volkswagen']
models = ['A4', 'X3', 'C-Class', 'Outlander', 'Clio', 'Camry', 'Golf']
body_types = ['crossover', 'hatch', 'other', 'sedan', 'vagon', 'van']
engine_types = ['Diesel', 'Gas', 'Other', 'Petrol']
registrations = ['yes', 'no']

# Generate random data
data = {
    'Brand': np.random.choice(brands, n_samples),
    'Model': np.random.choice(models, n_samples),
    'Year': np.random.randint(2000, 2024, n_samples),
    'Body': np.random.choice(body_types, n_samples),
    'Mileage': np.random.exponential(50000, n_samples).astype(int),
    'EngineV': np.random.normal(2.0, 0.8, n_samples),
    'Engine Type': np.random.choice(engine_types, n_samples),
    'Registration': np.random.choice(registrations, n_samples, p=[0.8, 0.2])
}

# Ensure realistic constraints
data['Mileage'] = np.clip(data['Mileage'], 1000, 300000)
data['EngineV'] = np.clip(data['EngineV'], 0.8, 6.0)

# Create a realistic price based on the features
def calculate_price(row):
    base_price = 15000
    
    # Brand multiplier
    brand_multipliers = {
        'BMW': 1.6, 'Mercedes-Benz': 1.7, 'Audi': 1.5,
        'Volkswagen': 1.2, 'Toyota': 1.3, 'Renault': 1.0, 'Mitsubishi': 1.1
    }
    
    # Year effect (newer cars are more expensive)
    year_effect = (row['Year'] - 2000) * 500
    
    # Mileage effect (higher mileage reduces price)
    mileage_effect = -row['Mileage'] * 0.1
    
    # Engine volume effect
    engine_effect = row['EngineV'] * 3000
    
    # Body type effect
    body_multipliers = {
        'sedan': 1.0, 'crossover': 1.2, 'hatch': 0.9,
        'vagon': 1.1, 'van': 1.3, 'other': 0.8
    }
    
    # Engine type effect
    engine_type_effects = {
        'Petrol': 0, 'Diesel': 2000, 'Gas': -1000, 'Other': -500
    }
    
    # Registration effect
    registration_effect = 2000 if row['Registration'] == 'yes' else 0
    
    price = (base_price * brand_multipliers.get(row['Brand'], 1.0) * 
             body_multipliers.get(row['Body'], 1.0) +
             year_effect + mileage_effect + engine_effect + 
             engine_type_effects.get(row['Engine Type'], 0) + 
             registration_effect)
    
    # Add some random noise
    price += np.random.normal(0, 2000)
    
    return max(price, 3000)  # Minimum price of $3,000

# Create DataFrame
df = pd.DataFrame(data)
df['Price'] = df.apply(calculate_price, axis=1)

# Round the price to nearest 100
df['Price'] = (df['Price'] / 100).round() * 100

# Add some missing values to simulate real data
missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
missing_columns = np.random.choice(['EngineV', 'Mileage'], size=len(missing_indices))
for i, col in zip(missing_indices, missing_columns):
    df.loc[i, col] = np.nan

# Save to CSV
df.to_csv('carproject.csv', index=False)

print(f"Generated {len(df)} car records")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nPrice statistics:")
print(df['Price'].describe())
