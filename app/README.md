# Commercial Real Estate Recommendation System

This Streamlit application helps businesses find the optimal location for their commercial real estate needs based on industry, region preferences, price range, and office size requirements.

## UI Mockup

The user interface consists of:

1. **Input Panel (Left Side)**:
   - Industry selection dropdown (Finance, Technology, Legal Services)
   - Recommended region (Northeast by default) with an explanation
   - Region selection dropdown (Northeast, South, Midwest, West)  
   - Price range selection dropdown (Very low, Low, Average, High, Very high)
   - Office size selection dropdown (Small, Mid Size, Large, Extra Large)
   - "Get Recommendations" button

2. **Results Panel (Right Side)**:
   - Initially shows a placeholder message
   - After clicking "Get Recommendations," displays:
     - Recommended city with key metrics:
       - Average Rent ($/SF)
       - Crime Rate (%)
       - Occupancy Rate (%)
       - Property Tax (%)
     - Market Analysis section with selected parameters
     - Alternative Recommendations with tabbed interface showing 3 alternative cities

## Features

- Integration with R models for city recommendations
- Beautiful UI with ShadCN components
- Dynamic updating of recommendations based on user inputs
- Responsive layout

## Running the Application

```bash
streamlit run app/realestate_app.py
```

## Dependencies

- streamlit
- pandas
- rpy2
- streamlit_shadcn_ui 