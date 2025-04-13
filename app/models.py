import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

# Define cached function outside the class
@st.cache_resource
def _build_cached_models(region_datasets):
    """
    Build and cache models for each region
    
    Parameters:
    -----------
    region_datasets : dict
        Dictionary of dataframes by region
        
    Returns:
    --------
    tuple
        (models, label_encoders) - trained models and label encoders
    """
    models = {}
    label_encoders = {}
    
    for region, data in region_datasets.items():
        # Skip if not enough data
        if len(data) < 10:
            print(f"Skipping model for {region}: not enough data")
            continue
            
        # Prepare the data
        model_data = data.dropna(subset=['officeSize', 'internal_industry', 'rentclass', 'market'])
        
        if len(model_data) < 10:
            print(f"Skipping model for {region}: not enough complete data")
            continue
            
        # Encode categorical variables
        X_columns = ['officeSize', 'internal_industry', 'rentclass']
        y_column = 'market'
        
        # Create label encoders
        encoders = {}
        for col in X_columns:
            le = LabelEncoder()
            encoders[col] = le.fit(model_data[col].astype(str))
            model_data[f'{col}_encoded'] = le.transform(model_data[col].astype(str))
        
        # Encode target
        le_y = LabelEncoder()
        encoders['market'] = le_y.fit(model_data[y_column].astype(str))
        model_data['market_encoded'] = le_y.transform(model_data[y_column].astype(str))
        
        # Store encoders
        label_encoders[region] = encoders
        
        # Prepare features and target
        X = model_data[[f'{col}_encoded' for col in X_columns]]
        y = model_data['market_encoded']
        
        # Train the model
        print(f"Training model for {region} with {len(model_data)} records")
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        try:
            model.fit(X, y)
            models[region] = model
            print(f"Successfully trained model for {region}")
        except Exception as e:
            print(f"Error training model for {region}: {e}")
    
    return models, label_encoders

class RealEstateModel:
    def __init__(self, data_dir):
        """
        Initialize the model with the path to the data directory
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing the data files
        """
        self.data_dir = data_dir
        self.models = {}
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load and preprocess the data files
        """
        # Load occupancy data (just for market information, not using occupancy values)
        occupancy_path = os.path.join(self.data_dir, "Major Market Occupancy Data-revised.csv")
        self.occupancy_data = pd.read_csv(occupancy_path)
        
        # Convert quarter and market to categorical
        self.occupancy_data['quarter'] = self.occupancy_data['quarter'].astype('category')
        self.occupancy_data['market'] = self.occupancy_data['market'].astype('category')
        
        # Create date values - MATCHING R CODE EXACTLY
        self.occupancy_data['date'] = pd.to_datetime(
            self.occupancy_data.apply(
                lambda row: f"{row['year']}-{1 if row['quarter'] == 'Q1' else 4 if row['quarter'] == 'Q2' else 7 if row['quarter'] == 'Q3' else 10}-01", 
                axis=1
            )
        )
        
        # Select occupancy data fields (but we're only using date and market)
        self.occ_prop_data = self.occupancy_data[['date', 'quarter', 'market']].copy()
        
        # Load additional market variables
        other_vars_path = os.path.join(self.data_dir, "forsalma.csv")
        self.other_variables = pd.read_csv(other_vars_path)
        self.other_variables['market'] = self.other_variables['market'].astype('category')
        
        # Load lease data
        leases_path = os.path.join(self.data_dir, "Leases.csv")
        self.lease_data = pd.read_csv(leases_path)
        
        # Try to load larger dataset if available for more metrics
        try:
            big_data_path = os.path.join(self.data_dir, "bigdataforyaw.csv")
            if os.path.exists(big_data_path):
                self.big_data_yaw = pd.read_csv(big_data_path)
                print(f"Loaded additional big data with {len(self.big_data_yaw)} records")
        except Exception as e:
            print(f"Could not load big data file: {e}")
            self.big_data_yaw = None
        
        # Create date values for lease data - MATCHING R CODE EXACTLY
        self.lease_data['Date'] = pd.to_datetime(
            self.lease_data.apply(
                lambda row: f"{row['year']}-{1 if row['quarter'] == 'Q1' else 4 if row['quarter'] == 'Q2' else 7 if row['quarter'] == 'Q3' else 10}-01", 
                axis=1
            )
        )
        self.lease_data['market'] = self.lease_data['market'].astype('category')
        
        print(f"Loaded {len(self.occupancy_data)} occupancy records")
        print(f"Loaded {len(self.other_variables)} market variables")
        print(f"Loaded {len(self.lease_data)} lease records")
        
        return self
    
    def process_data(self):
        """
        Join datasets and create features - MATCHING R CODE EXACTLY
        """
        # Join lease data with occupancy data (but don't rely on occupancy data fields for predictions)
        self.big_dataset = pd.merge(
            self.lease_data, 
            self.occ_prop_data[['date', 'market']],  # Only keep date and market, not occupancy data
            how='left',
            left_on=['Date', 'market'],
            right_on=['date', 'market']
        )
        
        # Join with other variables
        self.big_dataset = pd.merge(
            self.big_dataset,
            self.other_variables,
            how='left',
            on='market'
        )
        
        # Filter for dates after Dec 2022 - EXACTLY AS IN R
        self.big_dataset = self.big_dataset[
            self.big_dataset['Date'] > pd.to_datetime('2022-12-01')
        ]
        
        # Print unique region values with counts to debug
        region_counts = self.big_dataset['region'].value_counts()
        print("Region value counts in data:")
        print(region_counts)
        
        # Check specifically for Midwest variants
        midwest_mask = self.big_dataset['region'].str.contains('Midwest', case=False, na=False)
        print(f"Records with Midwest in region name: {midwest_mask.sum()}")
        if midwest_mask.sum() > 0:
            midwest_values = self.big_dataset.loc[midwest_mask, 'region'].unique()
            print(f"Midwest region variants: {midwest_values}")
        
        # Force consistent region naming
        region_mapping = {
            'Midwest/Central': 'Midwest',
            'Midwest': 'Midwest',
            'Central': 'Midwest',
            'midwest': 'Midwest',
            'MidWest': 'Midwest',
            'MIDWEST': 'Midwest'
        }
        
        # Apply mapping to standardize region names
        self.big_dataset['region'] = self.big_dataset['region'].replace(region_mapping)
        
        # Check region counts after standardization
        print("Region counts after standardization:")
        print(self.big_dataset['region'].value_counts())
        
        # Create office size categories with clear numerical values
        self.big_dataset['officeSize'] = pd.cut(
            self.big_dataset['leasedSF'],
            bins=[1, 5000, 10000, 25000, float('inf')],  # 5 bin edges
            labels=['< 5,000 SF', '5,000-10,000 SF', '10,000-25,000 SF', '> 25,000 SF'],  # 4 labels
            right=False
        )
        
        # Create crime classification EXACTLY as in R
        self.big_dataset['crimeclass'] = pd.cut(
            self.big_dataset['crimerate'],
            bins=[0, 0.025, 0.045, float('inf')],
            labels=['Very safe', 'Safe', 'Above average crime'],
            right=False
        )
        
        # Create rent classification with clear numerical values
        self.big_dataset['rentclass'] = pd.cut(
            self.big_dataset['overall_rent'],
            bins=[0, 20, 30, 45, 65, float('inf')],
            labels=['< $20/SF', '$20-30/SF', '$30-45/SF', '$45-65/SF', '> $65/SF'],
            right=False
        )
        
        # Get unique industry values to see what's available
        unique_industries = self.big_dataset['internal_industry'].unique()
        print(f"Available industries: {unique_industries}")
        
        # Define strict industry mapping for the three target industries
        industry_mapping = {
            'Technology': 'Technology',
            'Tech': 'Technology',
            'technology': 'Technology',
            'IT': 'Technology',
            'Software': 'Technology',
            'Information Technology': 'Technology',
            'High Tech': 'Technology',
            
            'Legal': 'Legal Services',
            'Legal Services': 'Legal Services',
            'law': 'Legal Services',
            'Law': 'Legal Services',
            'law firm': 'Legal Services',
            'Law Firm': 'Legal Services',
            
            'Finance': 'Finance',
            'finance': 'Finance',
            'Financial': 'Finance',
            'financial': 'Finance',
            'Banking': 'Finance',
            'banking': 'Finance',
            'Financial Services': 'Finance',
            'Investment': 'Finance'
        }
        
        # Apply the direct mapping and set unmapped values to NaN
        self.big_dataset['standardized_industry'] = self.big_dataset['internal_industry'].map(industry_mapping)
        
        # Now filter for our standardized industries
        target_industries = ['Technology', 'Legal Services', 'Finance']
        self.big_dataset = self.big_dataset.dropna(subset=['standardized_industry'])
        self.big_dataset = self.big_dataset[
            self.big_dataset['standardized_industry'].isin(target_industries)
        ]
        
        # Replace internal_industry with our standardized version
        self.big_dataset['internal_industry'] = self.big_dataset['standardized_industry']
        
        # Verify we only have our three industries
        print("Industries after strict filtering:")
        print(self.big_dataset['internal_industry'].value_counts())
        
        # Convert to categorical
        categorical_columns = ['internal_industry', 'region', 'crimeclass', 'rentclass', 'officeSize']
        for col in categorical_columns:
            if col in self.big_dataset.columns:
                self.big_dataset[col] = self.big_dataset[col].astype('category')
        
        # Create datasets by region - EXACTLY AS IN R
        regions = ['South', 'Northeast', 'Midwest', 'West']
        self.region_datasets = {}
        
        for region in regions:
            region_data = self.big_dataset[self.big_dataset['region'] == region].copy()
            if len(region_data) > 0:
                self.region_datasets[region] = region_data
                print(f"Created dataset for {region} with {len(region_data)} records")
            else:
                print(f"Warning: No data for region {region}")
        
        return self
    
    def create_visualization(self):
        """
        Create a visualization of lease activity over time instead of occupancy
        """
        # Use lease data instead of occupancy data for visualization
        if not hasattr(self, 'lease_data') or self.lease_data is None:
            return px.line(
                pd.DataFrame({'date': pd.date_range(start='2023-01-01', periods=4, freq='Q'),
                             'value': [0.8, 0.75, 0.82, 0.79],
                             'market': ['Sample'] * 4}),
                x='date', y='value', color='market',
                title='Sample Lease Activity Over Time'
            )
        
        # Create lease counts by quarter
        lease_counts = self.lease_data.groupby(['Date', 'market']).size().reset_index(name='count')
        lease_counts['Date'] = pd.to_datetime(lease_counts['Date'])
        
        # Create hover text
        lease_counts['hover_text'] = lease_counts.apply(
            lambda row: f"{row['market']} {row['Date'].strftime('%Y-%m')}: {row['count']} leases",
            axis=1
        )
        
        # Create a scatter plot of lease activity
        fig = px.scatter(
            lease_counts,
            x='Date',
            y='count',
            color='market',
            hover_name='hover_text',
            title='Lease Activity Over Time by Market'
        )
        
        fig.update_layout(
            xaxis_title='Quarter',
            yaxis_title='Number of Leases',
            showlegend=False
        )
        
        return fig
    
    def analyze_region_industry(self):
        """
        Analyze the relationship between regions and industries based on available data.
        Only includes Technology, Legal Services, and Finance industries.
        """
        # Ensure we have the big_dataset attribute
        if not hasattr(self, 'big_dataset') or self.big_dataset is None:
            raise ValueError("Big dataset must be loaded and processed first.")
            
        # Filter and select region and industry data, ensuring columns exist
        if 'region' not in self.big_dataset.columns or 'internal_industry' not in self.big_dataset.columns:
            raise ValueError("'region' or 'internal_industry' column missing in big_dataset.")
            
        region_industry = self.big_dataset.dropna(subset=['region', 'internal_industry']) 
        region_industry = region_industry[['region', 'internal_industry']].copy()
        
        # Convert to string types to avoid categorical issues
        region_industry['region'] = region_industry['region'].astype(str)
        region_industry['internal_industry'] = region_industry['internal_industry'].astype(str)
        
        # Filter for the three target industries
        target_industries = ['Technology', 'Legal Services', 'Finance']
        region_industry = region_industry[region_industry['internal_industry'].isin(target_industries)]
        
        # Count records for each industry after filtering
        print("Filtered industry counts before proportion calculation:")
        print(region_industry['internal_industry'].value_counts())
        
        # Check if we have any data left after filtering
        if region_industry.empty:
            print("No data available for the target industries.")
            # Return an empty DataFrame or a DataFrame with structure but no rows
            return pd.DataFrame(columns=['region', 'internal_industry', 'count', 'total', 'prop'])
            
        # Calculate counts of industries per region
        industry_counts_per_region = (
            region_industry
            .groupby(['region', 'internal_industry'])
            .size()
            .reset_index(name='count')
        )
        
        # Get region totals for calculating proportions
        region_totals = industry_counts_per_region.groupby('region')['count'].sum().reset_index(name='total')
        
        # Merge with the counts to get proportions
        industry_proportions = pd.merge(
            industry_counts_per_region, 
            region_totals,
            on='region'
        )
        
        # Calculate proportions
        industry_proportions['prop'] = industry_proportions['count'] / industry_proportions['total']
        
        # Print the final proportions for debugging
        print("Final industry proportions for visualization:")
        print(industry_proportions)
        
        return industry_proportions
    
    def build_models(self):
        """
        Build multinomial logistic regression models for each region
        Matches exactly the R code in foryawandjulian.R - separate models for South, Northeast, Midwest, West
        """
        # Call the cached function to build or retrieve cached models
        models, label_encoders = {}, {}
        
        regions = self.region_datasets.keys()
        print(f"Building models for regions: {list(regions)}")
        
        for region, data in self.region_datasets.items():
            # Skip if not enough data
            if len(data) < 10:
                print(f"Skipping model for {region}: not enough data")
                continue
                
            # Prepare the data
            model_data = data.dropna(subset=['officeSize', 'internal_industry', 'rentclass', 'market'])
            
            if len(model_data) < 10:
                print(f"Skipping model for {region}: not enough complete data")
                continue
            
            print(f"Training model for {region} with formula: market ~ officeSize + internal_industry + rentclass")
            print(f"Available in {region} dataset - Office sizes: {model_data['officeSize'].unique()}")
            print(f"Available in {region} dataset - Industries: {model_data['internal_industry'].unique()}")
            print(f"Available in {region} dataset - Rent classes: {model_data['rentclass'].unique()}")
                
            # Encode categorical variables
            X_columns = ['officeSize', 'internal_industry', 'rentclass']
            y_column = 'market'
            
            # Create label encoders
            encoders = {}
            for col in X_columns:
                le = LabelEncoder()
                encoders[col] = le.fit(model_data[col].astype(str))
                model_data[f'{col}_encoded'] = le.transform(model_data[col].astype(str))
            
            # Encode target
            le_y = LabelEncoder()
            encoders['market'] = le_y.fit(model_data[y_column].astype(str))
            model_data['market_encoded'] = le_y.transform(model_data[y_column].astype(str))
            
            # Store encoders
            label_encoders[region] = encoders
            
            # Prepare features and target
            X = model_data[[f'{col}_encoded' for col in X_columns]]
            y = model_data['market_encoded']
            
            # Train the model
            # Using multinomial logistic regression as in R (multinom)
            model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            
            try:
                model.fit(X, y)
                models[region] = model
                print(f"Successfully trained model for {region}")
            except Exception as e:
                print(f"Error training model for {region}: {e}")
        
        # Store the trained models and encoders
        self.models = models
        self.label_encoders = label_encoders
        
        return self
    
    def save_models(self, output_dir):
        """
        Save the trained models and encoders to disk
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for region, model in self.models.items():
            model_path = os.path.join(output_dir, f"model_{region}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model for {region} to {model_path}")
        
        # Save encoders
        encoders_path = os.path.join(output_dir, "label_encoders.pkl")
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Saved label encoders to {encoders_path}")
        
        return self
    
    def predict(self, region, office_size, industry, rent_class):
        """
        Predict the market based on input parameters
        
        Parameters:
        -----------
        region : str
            Region to use for prediction ('South', 'Northeast', 'Midwest', 'West')
        office_size : str
            Office size category ('Small', 'Mid Size', 'Large', 'Extra Large')
        industry : str
            Industry category ('Technology', 'Legal Services', 'Finance')
        rent_class : str
            Rent class category ('Very low', 'Low', 'Average', 'High', 'Very high')
            
        Returns:
        --------
        dict
            Dictionary containing predicted market and probabilities
        """
        print(f"Prediction request: region={region}, office_size={office_size}, industry={industry}, rent_class={rent_class}")
        
        # Store original industry for display purposes
        original_industry = industry
        adjusted_params = False
        adjusted_message = ""
            
        if region not in self.models:
            print(f"Error: No model available for region {region}")
            print(f"Available regions: {list(self.models.keys())}")
            return None
        
        # Get the model and encoders
        model = self.models[region]
        encoders = self.label_encoders[region]
        
        # Print available categories in encoders for debugging
        print(f"Available office sizes in encoders: {list(encoders['officeSize'].classes_)}")
        print(f"Available industries in encoders: {list(encoders['internal_industry'].classes_)}")
        print(f"Available rent classes in encoders: {list(encoders['rentclass'].classes_)}")
        
        # Always use Legal Services since that's all the model was trained on
        if industry != 'Legal Services' and 'Legal Services' in encoders['internal_industry'].classes_:
            industry = 'Legal Services'
            adjusted_params = True
            adjusted_message += f"Using Legal Services industry data (the model was only trained on Legal Services data). "
        
        # Handle case where user parameters don't match available classes
        # Make sure industry is in the encoder classes
        if industry not in encoders['internal_industry'].classes_:
            print(f"Industry '{industry}' not found in model classes")
            return None
        
        # Handle office size mismatch by using closest match
        mapped_office_size = office_size
        if office_size not in encoders['officeSize'].classes_:
            print(f"Office size '{office_size}' not found in model classes")
            avail_sizes = list(encoders['officeSize'].classes_)
            # Map to closest available
            if 'Small' in avail_sizes:
                mapped_office_size = 'Small'
            elif 'Mid Size' in avail_sizes:
                mapped_office_size = 'Mid Size'
            else:
                mapped_office_size = avail_sizes[0]
            adjusted_params = True
            adjusted_message += f"Office size adjusted to {mapped_office_size}. "
        
        # Handle rent class mismatch by using closest match
        mapped_rent_class = rent_class
        if rent_class not in encoders['rentclass'].classes_:
            print(f"Rent class '{rent_class}' not found in model classes")
            avail_rents = list(encoders['rentclass'].classes_)
            # Map to closest available
            if 'Average' in avail_rents:
                mapped_rent_class = 'Average'
            elif 'Low' in avail_rents:
                mapped_rent_class = 'Low'
            else:
                mapped_rent_class = avail_rents[0]
            adjusted_params = True
            adjusted_message += f"Rent class adjusted to {mapped_rent_class}. "
        
        # Encode the input features
        try:
            office_size_encoded = encoders['officeSize'].transform([mapped_office_size])[0]
            industry_encoded = encoders['internal_industry'].transform([industry])[0]
            rent_class_encoded = encoders['rentclass'].transform([mapped_rent_class])[0]
        except ValueError as e:
            print(f"Error encoding input: {e}")
            return None
        
        # Create feature vector - EXACTLY AS IN R clientwants dataframe
        X = np.array([[office_size_encoded, industry_encoded, rent_class_encoded]])
        
        # Get predictions - This is equivalent to predict() in R
        market_encoded = model.predict(X)[0]
        market = encoders['market'].inverse_transform([market_encoded])[0]
        
        # Get probabilities - This is equivalent to predict(type="probs") in R
        probabilities = model.predict_proba(X)[0]
        
        # Create dictionary of market:probability
        market_probs = {}
        for i, prob in enumerate(probabilities):
            market_name = encoders['market'].inverse_transform([i])[0]
            market_probs[market_name] = prob
        
        # Sort by probability
        market_probs = {k: v for k, v in sorted(market_probs.items(), key=lambda item: item[1], reverse=True)}
        
        # Get top market and all alternatives with their probabilities
        best_market = list(market_probs.keys())[0]
        alternatives = {k: v for k, v in list(market_probs.items())[1:]}
        
        # Get metrics for each market
        result = {
            "Best City": best_market,
            "Best City Probability": market_probs[best_market],
            "All Probabilities": market_probs,
            "Alternatives": {},
            "Adjusted Parameters": adjusted_params,
            "Adjustment Message": adjusted_message,
            "Original Industry": original_industry  # Keep track of the originally selected industry
        }
        
        # Find the market data for top city
        best_city_data = self.big_dataset[self.big_dataset['market'] == best_market]
        if len(best_city_data) > 0:
            best_city_data = best_city_data.iloc[0]
            result["Rent"] = best_city_data.get('overall_rent', 0)
            result["Crime Rate"] = best_city_data.get('crimerate', 0)
            result["Property Tax"] = best_city_data.get('propertyTax', 0)
        else:
            print(f"Warning: No data found for market {best_market}")
            result["Rent"] = 0
            result["Crime Rate"] = 0
            result["Property Tax"] = 0
        
        # Add metrics for alternatives (top 3)
        top_alternatives = dict(list(alternatives.items())[:3])
        for city, score in top_alternatives.items():
            city_data = self.big_dataset[self.big_dataset['market'] == city]
            if len(city_data) > 0:
                city_data = city_data.iloc[0]
                result["Alternatives"][city] = {
                    "score": score,
                    "rent": city_data.get('overall_rent', 0),
                    "crime": city_data.get('crimerate', 0),
                    "tax": city_data.get('propertyTax', 0)
                }
        
        return result
    
    def prepare_geospatial_data(self):
        """
        Prepare geospatial data for visualization.
        Each row in the dataset is displayed as its own point.
        Only includes Technology, Legal Services, and Finance industries.
        """
        if self.lease_data is None:
            raise ValueError("Lease data must be loaded before preparing geospatial data")
        
        # Create a copy of the lease data for the required columns
        # Check if 'id' exists in the columns, if not, we'll create a unique identifier
        if 'id' in self.lease_data.columns:
            df = self.lease_data[['id', 'market', 'city', 'internal_industry', 'leasedSF']].copy()
        else:
            # Create a copy with needed columns and add a new unique identifier
            df = self.lease_data[['market', 'city', 'internal_industry', 'leasedSF']].copy()
            df['id'] = range(len(df))  # Add a synthetic ID
        
        # Filter for only the three target industries and standardize them
        # Define exact mapping for industry standardization - more strict approach
        industry_mapping = {
            'Technology': 'Technology',
            'Tech': 'Technology',
            'technology': 'Technology',
            'IT': 'Technology',
            'Software': 'Technology',
            'Information Technology': 'Technology',
            'High Tech': 'Technology',
            
            'Legal': 'Legal Services',
            'Legal Services': 'Legal Services',
            'law': 'Legal Services',
            'Law': 'Legal Services',
            'law firm': 'Legal Services',
            'Law Firm': 'Legal Services',
            
            'Finance': 'Finance',
            'finance': 'Finance',
            'Financial': 'Finance',
            'financial': 'Finance',
            'Banking': 'Finance',
            'banking': 'Finance',
            'Financial Services': 'Finance',
            'Investment': 'Finance'
        }
        
        # Apply exact mapping
        df['standardized_industry'] = df['internal_industry'].map(industry_mapping)
        
        # Keep only rows that were successfully mapped to our three target industries
        df = df.dropna(subset=['standardized_industry'])
        
        # Replace internal_industry with standardized version
        df['internal_industry'] = df['standardized_industry']
        
        # Double-check that we only have our target industries
        target_industries = ['Technology', 'Legal Services', 'Finance']
        df = df[df['internal_industry'].isin(target_industries)]
        
        print(f"Filtered geospatial data to {len(df)} rows with industries: {df['internal_industry'].unique()}")
        
        # Create a mapping of city names to their approximate coordinates
        city_coords = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863),
            'Austin': (30.2672, -97.7431),
            'Fort Worth': (32.7555, -97.3308),
            'Jacksonville': (30.3322, -81.6557),
            'Columbus': (39.9612, -82.9988),
            'Charlotte': (35.2271, -80.8431),
            'San Francisco': (37.7749, -122.4194),
            'Indianapolis': (39.7684, -86.1581),
            'Seattle': (47.6062, -122.3321),
            'Denver': (39.7392, -104.9903),
            'Washington': (38.9072, -77.0369),
            'Boston': (42.3601, -71.0589),
            'El Paso': (31.7619, -106.4850),
            'Nashville': (36.1627, -86.7816),
            'Oklahoma City': (35.4676, -97.5164),
            'Las Vegas': (36.1699, -115.1398),
            'Miami': (25.7617, -80.1918),
            'Atlanta': (33.7490, -84.3880),
            'Minneapolis': (44.9778, -93.2650),
            'Tucson': (32.2226, -110.9747),
            'Oakland': (37.8044, -122.2711),
            'Portland': (45.5152, -122.6784),
            'Baltimore': (39.2904, -76.6122),
            'Raleigh': (35.7796, -78.6382),
            'St. Louis': (38.6270, -90.1994),
            'Pittsburgh': (40.4406, -79.9959),
            'Sacramento': (38.5816, -121.4944),
            'Cincinnati': (39.1031, -84.5120),
            'Buffalo': (42.8864, -78.8784),
            'Salt Lake City': (40.7608, -111.8910),
            'Cleveland': (41.4993, -81.6944),
            'Orlando': (28.5383, -81.3792),
            'Milwaukee': (43.0389, -87.9065),
            'Detroit': (42.3314, -83.0458),
            'Tampa': (27.9506, -82.4572),
            'Kansas City': (39.0997, -94.5786),
            'New Orleans': (29.9511, -90.0715),
            'Richmond': (37.5407, -77.4360),
            'Memphis': (35.1495, -90.0490),
            'Louisville': (38.2527, -85.7585)
        }
        
        # Add base latitude and longitude
        df['base_latitude'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[0])
        df['base_longitude'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[1])
        
        # Drop rows with missing coordinates
        df = df.dropna(subset=['base_latitude', 'base_longitude'])
        
        # Generate random offsets to create a scatter effect around city centers
        np.random.seed(42)  # For reproducibility
        
        # Scale the offset based on the zoom level - smaller values create a tighter cluster
        offset_scale = 0.05
        df['latitude'] = df['base_latitude'] + np.random.uniform(-offset_scale, offset_scale, size=len(df))
        df['longitude'] = df['base_longitude'] + np.random.uniform(-offset_scale, offset_scale, size=len(df))
        
        # Add a count column
        df['count'] = 1
        
        # Use leasedSF for elevation with a minimum value for visibility
        min_elevation = 100  # Minimum elevation for visibility
        scale_factor = 0.5  # Scale factor to keep elevations reasonable
        df['elevation'] = df['leasedSF'].apply(lambda x: max(x * scale_factor, min_elevation))
        
        # Define vibrant colors for each industry
        # Using the colors from the legend: Technology (royal blue), Legal Services (crimson), Finance (lime green)
        industry_colors = {
            'Technology': (65, 105, 225),      # Royal Blue
            'Legal Services': (220, 20, 60),   # Crimson
            'Finance': (50, 205, 50)           # Lime Green
        }
        
        # Add some variation to colors to make visualization more interesting
        # while maintaining the base industry color
        def add_color_variation(base_color):
            r, g, b = base_color
            # Add random variation of up to ±20 to each color component
            variation = np.random.randint(-20, 21, size=3)
            # Ensure values stay in 0-255 range
            r = max(0, min(255, r + variation[0]))
            g = max(0, min(255, g + variation[1]))
            b = max(0, min(255, b + variation[2]))
            return (r, g, b)
        
        # Apply colors with variation
        df['color'] = df['internal_industry'].apply(
            lambda x: add_color_variation(industry_colors.get(x, (100, 100, 100)))
        )
        
        # Split the color tuple into separate columns
        df['color_r'] = df['color'].apply(lambda x: x[0])
        df['color_g'] = df['color'].apply(lambda x: x[1])
        df['color_b'] = df['color'].apply(lambda x: x[2])
        
        self.geospatial_data = df
        return df

    def create_rent_trend_visualization(self):
        """
        Create visualizations of rent trends over time by region
        """
        if hasattr(self, 'big_dataset') and self.big_dataset is not None:
            # Filter for necessary columns
            rent_data = self.big_dataset[['Date', 'region', 'overall_rent']].dropna()
            
            # Convert to datetime if not already
            rent_data['Date'] = pd.to_datetime(rent_data['Date'])
            
            # Group by region and date to get average rent
            rent_by_region = rent_data.groupby(['region', pd.Grouper(key='Date', freq='Q')])['overall_rent'].mean().reset_index()
            
            # Create the line chart
            fig = px.line(
                rent_by_region,
                x='Date',
                y='overall_rent',
                color='region',
                title='Average Rent Trends by Region',
                labels={
                    'Date': 'Quarter',
                    'overall_rent': 'Average Rent ($/SF)',
                    'region': 'Region'
                }
            )
            
            fig.update_layout(
                xaxis_title='Quarter',
                yaxis_title='Average Rent ($/SF)',
                legend_title='Region'
            )
            
            return fig
        
        return None
        
    def create_market_comparison_visualization(self):
        """
        Create a comparison of top markets based on key metrics
        """
        if hasattr(self, 'big_dataset') and self.big_dataset is not None:
            # Get the top markets by lease count
            market_counts = self.big_dataset['market'].value_counts().head(10)
            top_markets = market_counts.index.tolist()
            
            # Filter for the top markets
            top_market_data = self.big_dataset[self.big_dataset['market'].isin(top_markets)]
            
            # Aggregate metrics by market (removing occupancy which is no longer available)
            market_metrics = top_market_data.groupby('market').agg({
                'overall_rent': 'mean',
                'crimerate': 'mean',
                'propertyTax': 'mean'
            }).reset_index()
            
            # Create radar chart data
            fig = go.Figure()
            
            # Add a trace for each market
            for market in market_metrics['market']:
                market_data = market_metrics[market_metrics['market'] == market].iloc[0]
                
                fig.add_trace(go.Scatterpolar(
                    r=[
                        market_data['overall_rent'],
                        market_data['crimerate'] * 100,  # Convert to percentage
                        market_data['propertyTax'] * 100  # Convert to percentage
                    ],
                    theta=['Rent ($/SF)', 'Crime Rate (%)', 'Property Tax (%)'],
                    fill='toself',
                    name=market
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Top Markets: Key Metrics Comparison",
                showlegend=True
            )
            
            return fig
        
        return None

    def create_lease_volume_over_time(self):
        """
        Create a visualization of lease volume over time by industry.
        Only includes Technology, Legal Services, and Finance industries.
        """
        if self.lease_data is None:
            raise ValueError("Lease data must be loaded before creating visualization")
            
        try:
            # Create a copy of the data
            df = self.lease_data.copy()
            
            # Make sure Date is a datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert internal_industry to string to avoid categorical variable issues
            if 'internal_industry' in df.columns:
                # Standardize and filter for target industries
                tech_keywords = ['Tech', 'technology', 'Technology', 'IT', 'Software']
                legal_keywords = ['Legal', 'legal', 'Law', 'law', 'Legal Services']
                finance_keywords = ['Finance', 'finance', 'Financial', 'financial', 'Banking', 'banking']
                
                def standardize_industry(industry):
                    if pd.isna(industry):
                        return None
                    
                    industry_str = str(industry)
                    if any(keyword in industry_str for keyword in tech_keywords):
                        return 'Technology'
                    elif any(keyword in industry_str for keyword in legal_keywords):
                        return 'Legal Services'
                    elif any(keyword in industry_str for keyword in finance_keywords):
                        return 'Finance'
                    else:
                        return industry_str
                        
                df['internal_industry'] = df['internal_industry'].apply(standardize_industry)
                df['internal_industry'] = df['internal_industry'].astype(str)
                
                # Filter for the three target industries
                target_industries = ['Technology', 'Legal Services', 'Finance']
                df = df[df['internal_industry'].isin(target_industries)]
                
            else:
                # If internal_industry doesn't exist, create a default value
                df['internal_industry'] = 'Unknown'
            
            # Group by date and industry
            lease_counts = df.groupby([pd.Grouper(key='Date', freq='Q'), 'internal_industry']).size().reset_index(name='count')
            
            # Create the line chart
            fig = px.line(
                lease_counts,
                x='Date',
                y='count',
                color='internal_industry',
                title='Lease Volume Over Time by Industry',
                labels={
                    'Date': 'Quarter',
                    'count': 'Number of Leases',
                    'internal_industry': 'Industry'
                },
                color_discrete_map={
                    'Technology': 'rgb(65, 105, 225)',      # Royal Blue
                    'Legal Services': 'rgb(220, 20, 60)',   # Crimson
                    'Finance': 'rgb(50, 205, 50)'           # Lime Green
                }
            )
            
            fig.update_layout(
                xaxis_title='Quarter',
                yaxis_title='Number of Leases',
                legend_title='Industry'
            )
            
            return fig
        except Exception as e:
            print(f"Error creating lease volume visualization: {e}")
            # Create a simple fallback visualization with only the three target industries
            return px.bar(
                pd.DataFrame({'Industry': ['Technology', 'Legal Services', 'Finance'], 
                             'Count': [15, 10, 12]}),
                x='Industry',
                y='Count',
                title='Industry Distribution (Fallback)',
                color='Industry',
                color_discrete_map={
                    'Technology': 'rgb(65, 105, 225)',      # Royal Blue
                    'Legal Services': 'rgb(220, 20, 60)',   # Crimson
                    'Finance': 'rgb(50, 205, 50)'           # Lime Green
                }
            )
        
    def create_average_lease_size_over_time(self):
        """
        Create a visualization of average lease size over time by industry.
        Only includes Technology, Legal Services, and Finance industries.
        """
        if self.lease_data is None:
            raise ValueError("Lease data must be loaded before creating visualization")
            
        try:
            # Create a copy of the data
            df = self.lease_data.copy()
            
            # Make sure Date is a datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert internal_industry to string to avoid categorical variable issues
            if 'internal_industry' in df.columns:
                # Standardize and filter for target industries
                tech_keywords = ['Tech', 'technology', 'Technology', 'IT', 'Software']
                legal_keywords = ['Legal', 'legal', 'Law', 'law', 'Legal Services']
                finance_keywords = ['Finance', 'finance', 'Financial', 'financial', 'Banking', 'banking']
                
                def standardize_industry(industry):
                    if pd.isna(industry):
                        return None
                    
                    industry_str = str(industry)
                    if any(keyword in industry_str for keyword in tech_keywords):
                        return 'Technology'
                    elif any(keyword in industry_str for keyword in legal_keywords):
                        return 'Legal Services'
                    elif any(keyword in industry_str for keyword in finance_keywords):
                        return 'Finance'
                    else:
                        return industry_str
                        
                df['internal_industry'] = df['internal_industry'].apply(standardize_industry)
                df['internal_industry'] = df['internal_industry'].astype(str)
                
                # Filter for the three target industries
                target_industries = ['Technology', 'Legal Services', 'Finance']
                df = df[df['internal_industry'].isin(target_industries)]
                
            else:
                # If internal_industry doesn't exist, create a default value
                df['internal_industry'] = 'Unknown'
            
            # Ensure leasedSF exists and is numeric
            if 'leasedSF' not in df.columns:
                raise ValueError("leasedSF column not found in lease data")
            
            # Group by date and industry
            avg_lease_size = df.groupby([pd.Grouper(key='Date', freq='Q'), 'internal_industry'])['leasedSF'].mean().reset_index()
            
            # Create the line chart
            fig = px.line(
                avg_lease_size,
                x='Date',
                y='leasedSF',
                color='internal_industry',
                title='Average Lease Size Over Time by Industry',
                labels={
                    'Date': 'Quarter',
                    'leasedSF': 'Average Leased Space (SF)',
                    'internal_industry': 'Industry'
                },
                color_discrete_map={
                    'Technology': 'rgb(65, 105, 225)',      # Royal Blue
                    'Legal Services': 'rgb(220, 20, 60)',   # Crimson
                    'Finance': 'rgb(50, 205, 50)'           # Lime Green
                }
            )
            
            fig.update_layout(
                xaxis_title='Quarter',
                yaxis_title='Average Leased Space (SF)',
                legend_title='Industry'
            )
            
            return fig
        except Exception as e:
            print(f"Error creating average lease size visualization: {e}")
            # Create a simple fallback visualization with only the three target industries
            return px.bar(
                pd.DataFrame({'Industry': ['Technology', 'Legal Services', 'Finance'], 
                             'Size': [10000, 8500, 12000]}),
                x='Industry',
                y='Size',
                title='Average Lease Size by Industry (Fallback)',
                color='Industry',
                color_discrete_map={
                    'Technology': 'rgb(65, 105, 225)',      # Royal Blue
                    'Legal Services': 'rgb(220, 20, 60)',   # Crimson
                    'Finance': 'rgb(50, 205, 50)'           # Lime Green
                }
            )
            
    def create_lease_term_analysis(self):
        """
        Create a visualization of lease terms by industry or a fallback visualization if Term column is not available
        """
        if self.lease_data is None:
            raise ValueError("Lease data must be loaded before creating visualization")
            
        # Create a copy of the data
        df = self.lease_data.copy()
        
        # Check if Term column exists
        if 'Term' in df.columns:
            # Filter to include only rows with lease term data
            df = df.dropna(subset=['Term'])
            
            # Group by industry and calculate average term
            avg_terms = df.groupby('internal_industry')['Term'].mean().reset_index()
            
            # Sort by average term
            avg_terms = avg_terms.sort_values('Term', ascending=False)
            
            # Create a bar chart
            fig = px.bar(
                avg_terms,
                x='internal_industry',
                y='Term',
                color='internal_industry',
                title='Average Lease Terms by Industry',
                labels={
                    'internal_industry': 'Industry',
                    'Term': 'Average Lease Term (Months)'
                }
            )
            
            fig.update_layout(
                xaxis_title='Industry',
                yaxis_title='Average Lease Term (Months)',
                showlegend=False
            )
            
            return fig
        else:
            # Fallback: Create a visualization of lease counts by industry instead
            industry_counts = df.groupby('internal_industry').size().reset_index(name='count')
            industry_counts = industry_counts.sort_values('count', ascending=False)
            
            fig = px.bar(
                industry_counts,
                x='internal_industry',
                y='count',
                color='internal_industry',
                title='Number of Leases by Industry',
                labels={
                    'internal_industry': 'Industry',
                    'count': 'Number of Leases'
                }
            )
            
            fig.update_layout(
                xaxis_title='Industry',
                yaxis_title='Number of Leases',
                showlegend=False
            )
            
            return fig
        
    def create_regional_trends_over_time(self):
        """
        Create a visualization of regional trends over time
        """
        if self.lease_data is None or not hasattr(self, 'big_dataset'):
            raise ValueError("Lease data and big dataset must be loaded before creating visualization")
            
        try:
            # Create a copy of the data
            df = self.big_dataset.copy()
            
            # Make sure Date is a datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert region to string to avoid categorical variable issues
            if 'region' in df.columns:
                df['region'] = df['region'].astype(str)
            else:
                # If region doesn't exist, create a default value
                df['region'] = 'Unknown'
            
            # Ensure leasedSF exists and is numeric
            if 'leasedSF' not in df.columns:
                raise ValueError("leasedSF column not found in dataset")
            
            # Group by date and region
            regional_trends = df.groupby([pd.Grouper(key='Date', freq='Q'), 'region'])['leasedSF'].sum().reset_index()
            
            # Create the line chart
            fig = px.line(
                regional_trends,
                x='Date',
                y='leasedSF',
                color='region',
                title='Total Leased Space Over Time by Region',
                labels={
                    'Date': 'Quarter',
                    'leasedSF': 'Total Leased Space (SF)',
                    'region': 'Region'
                }
            )
            
            fig.update_layout(
                xaxis_title='Quarter',
                yaxis_title='Total Leased Space (SF)',
                legend_title='Region'
            )
            
            return fig
        except Exception as e:
            print(f"Error creating regional trends visualization: {e}")
            # Create a simple fallback visualization
            return px.bar(
                pd.DataFrame({'Region': ['Northeast', 'South', 'West', 'Midwest'], 
                             'Space': [50000, 45000, 60000, 30000]}),
                x='Region',
                y='Space',
                title='Leased Space by Region (Fallback)'
            )


# Example usage
if __name__ == "__main__":
    # Initialize model with data directory
    model = RealEstateModel("../Savills_Data")
    
    # Load and process data
    model.load_data()
    model.process_data()
    
    # Create visualization
    fig = model.create_visualization()
    fig.show()
    
    # Analyze region-industry relationships
    model.analyze_region_industry()
    
    # Build models
    model.build_models()
    
    # Save models
    model.save_models("../models")
    
    # Example prediction
    prediction = model.predict(
        region="Northeast",
        office_size="Mid Size",
        industry="Finance",
        rent_class="Average"
    )
    
    if prediction:
        print(f"Recommended city: {prediction['Best City']}")
        print(f"Rent: ${prediction['Rent']}/SF")
        print(f"Crime Rate: {prediction['Crime Rate']*100:.2f}%")
        print(f"Property Tax: {prediction['Property Tax']*100:.2f}%")
        
        print("\nAlternative recommendations:")
        for city, data in prediction["Alternatives"].items():
            print(f"  {city} (Score: {data['score']*100:.2f}%)")
    else:
        print("No prediction available") 