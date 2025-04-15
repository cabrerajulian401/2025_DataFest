from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ✅ Allowed industries for dropdown
ALLOWED_INDUSTRIES = [
    "Technology, Advertising, Media, and Information",
    "Legal Services",
    "Financial Services and Insurance"
]

# 🗺️ City coordinates
city_coords = {
    'Atlanta': (33.7490, -84.3880),
    'Austin': (30.2672, -97.7431),
    'Baltimore': (39.2904, -76.6122),
    'Boston': (42.3601, -71.0589),
    'Charlotte': (35.2271, -80.8431),
    'Chicago': (41.8781, -87.6298),
    'Chicago Suburbs': (41.8119, -87.6873),
    'Dallas/Ft Worth': (32.7767, -96.7970),
    'Denver': (39.7392, -104.9903),
    'Detroit': (42.3314, -83.0458),
    'Houston': (29.7604, -95.3698),
    'Los Angeles': (34.0522, -118.2437),
    'Manhattan': (40.7831, -73.9712),
    'Nashville': (36.1627, -86.7816),
    'Northern New Jersey': (40.7357, -74.1724),
    'Northern Virginia': (38.8048, -77.0469),
    'Orange County': (33.7175, -117.8311),
    'Philadelphia': (39.9526, -75.1652),
    'Phoenix': (33.4484, -112.0740),
    'Raleigh/Durham': (35.7796, -78.6382),
    'Salt Lake City': (40.7608, -111.8910),
    'San Diego': (32.7157, -117.1611),
    'San Francisco': (37.7749, -122.4194),
    'Seattle': (47.6062, -122.3321),
    'South Bay/San Jose': (37.3382, -121.8863),
    'South Florida': (26.1224, -80.1373),
    'Southern Maryland': (38.3000, -76.6000),
    'Tampa': (27.9506, -82.4572),
    'Washington D.C.': (38.9072, -77.0369)
}

# Define the path to the CSV file
CSV_PATH = os.path.join(os.path.dirname(__file__), 'bigdataforyaw.csv')

# Cache the dataframe to avoid reading it on every request
df = None

def load_data():
    global df
    if df is None:
        # Read CSV with low_memory=False to avoid the mixed types warning
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
    return df

@app.route('/api/city-density')
def city_density():
    try:
        # Load data if not already loaded
        data_df = load_data()

        # ✅ Get industry from query param
        industry = request.args.get('industry')
        if industry not in ALLOWED_INDUSTRIES:
            return jsonify({"error": "Invalid or missing industry"}), 400

        # Filter dataset by industry
        filtered = data_df[data_df['internal_industry'] == industry]
        counts = filtered['market'].value_counts()

        data = []
        for market, count in counts.items():
            clean_market = market.strip()
            if clean_market in city_coords:
                lat, lon = city_coords[clean_market]
                # Adjust size calculation for better visualization
                size = (count ** 0.7) * 0.8
                data.append({
                    "market": clean_market,
                    "lat": lat,
                    "lon": lon,
                    "count": int(count),
                    "size": round(size, 3)
                })

        print(f"Returning {len(data)} locations for industry: {industry}")
        return jsonify(data)

    except Exception as e:
        print(f"Error in city_density: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    # Preload data when starting the server
    load_data()
    app.run(debug=True, port=5001)