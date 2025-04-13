import streamlit as st
import os
import pandas as pd
import plotly.express as px
import sys
import pydeck as pdk
from models import RealEstateModel
import streamlit.components.v1 as components

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Path to the data directory
data_dir = os.path.join(parent_dir, "Savills_Data")
models_dir = os.path.join(parent_dir, "models")

# Create a model instance
model = None

def load_model():
    global model
    if model is None:
        with st.spinner('Loading data and building models...'):
            model = RealEstateModel(data_dir)
            model.load_data()
            model.process_data()
            model.build_models()
    return model

# Page configuration
st.set_page_config(
    page_title="Commercial Real Estate Recommendation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with instructions
with st.sidebar:
    st.title("Commercial Real Estate Recommendation")
    st.markdown("""
    This application helps you find the perfect city for your commercial real estate needs.
    
    **How to use:**
    1. Select your region of interest
    2. Enter your office size requirements
    3. Choose your industry
    4. Pick a rent class preference
    5. Get personalized city recommendations
    
    The recommendations are based on a model trained with historical commercial real estate data.
    """)
    
    # Add credits
    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and Plotly")

# Main content area
st.title("Commercial Real Estate Recommendation System")

# Initialize model
model = load_model()

# Create tabs
tab1, tab2 = st.tabs(["Market Trends", "City Recommendations"])

with tab1:
    st.header("Market Trends Analysis")
    
    # Create expandable sections instead of tabs
    with st.expander("Occupancy & Rent Trends", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Display occupancy trends
            st.subheader("Occupancy Trends Over Time")
            fig = model.create_visualization()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display rent trends by region
            st.subheader("Rent Trends by Region")
            rent_fig = model.create_rent_trend_visualization()
            if rent_fig:
                st.plotly_chart(rent_fig, use_container_width=True)
            else:
                st.info("Rent trend data not available.")
    
    with st.expander("Lease Activity Trends", expanded=True):
        st.subheader("Lease Volume & Size Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display lease volume over time by industry
            lease_volume_fig = model.create_lease_volume_over_time()
            st.plotly_chart(lease_volume_fig, use_container_width=True)
            
            st.markdown("""
            **Key Insights:**
            - Track how different industries' leasing activity fluctuates over time
            - Identify seasonal patterns in leasing volume
            - Spot emerging trends in industry demand
            """)
        
        with col2:
            # Display average lease size over time by industry
            lease_size_fig = model.create_average_lease_size_over_time()
            st.plotly_chart(lease_size_fig, use_container_width=True)
            
            st.markdown("""
            **Key Insights:**
            - See how space requirements evolve by industry
            - Compare lease size variations across quarters
            - Identify which industries are expanding or contracting
            """)
        
        # Display lease term analysis
        st.subheader("Lease Term Analysis by Industry")
        lease_term_fig = model.create_lease_term_analysis()
        st.plotly_chart(lease_term_fig, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - Compare typical lease durations across different industries
        - Identify which sectors prefer longer-term commitments
        - Inform negotiation strategies based on industry norms
        """)
    
    with st.expander("Regional Market Insights", expanded=True):
        st.subheader("Regional Market Trends")
        
        # Display regional trends over time
        regional_trends_fig = model.create_regional_trends_over_time()
        st.plotly_chart(regional_trends_fig, use_container_width=True)
        
        # Add market comparison visualization (radar chart)
        st.subheader("Top Markets Comparison")
        market_fig = model.create_market_comparison_visualization()
        if market_fig:
            st.plotly_chart(market_fig, use_container_width=True)
            st.markdown("""
            **Chart Explanation:**
            - Each axis represents a different metric
            - Each market is plotted as a unique shape
            - Larger area indicates higher overall values
            - Compare markets across multiple metrics at once
            """)
        else:
            st.info("Market comparison data not available.")
    
    with st.expander("Industry Distribution", expanded=True):
        # Add industry trends across regions (treemap)
        st.subheader("Industry Distribution by Region")
        
        with st.spinner("Preparing industry visualization..."):
            # Create a DataFrame with industry data
            industry_data = model.big_dataset.dropna(subset=['internal_industry', 'region', 'leasedSF'])
            
            # Convert categorical columns to strings to avoid categorical ordering issues
            industry_data['region'] = industry_data['region'].astype(str)
            industry_data['internal_industry'] = industry_data['internal_industry'].astype(str)
            
            # Filter for only the three target industries
            target_industries = ['Technology', 'Legal Services', 'Finance']
            industry_data = industry_data[industry_data['internal_industry'].isin(target_industries)]
            
            # Group by region and industry to calculate total leased space
            industry_space = industry_data.groupby(['region', 'internal_industry'])['leasedSF'].sum().reset_index()
            
            # Create custom color mapping that matches our other visualizations
            color_map = {
                'Technology': 'rgb(65, 105, 225)',      # Royal Blue
                'Legal Services': 'rgb(220, 20, 60)',   # Crimson
                'Finance': 'rgb(50, 205, 50)',          # Lime Green
                'Northeast': '#66CCEE',
                'South': '#EE7733',
                'West': '#CCBB44',
                'Midwest': '#AA3377'
            }
            
            # Create a treemap visualization
            treemap_fig = px.treemap(
                industry_space,
                path=[px.Constant("All Regions"), 'region', 'internal_industry'],
                values='leasedSF',
                color='internal_industry',  # Color by industry instead of region
                title='Total Leased Space by Region and Industry',
                hover_data=['leasedSF'],
                color_discrete_map=color_map
            )
            
            treemap_fig.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                height=500
            )
            
            st.plotly_chart(treemap_fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **Treemap Insights:**
            - Boxes are sized by total leased square footage
            - Colors represent different industries:
              - Blue: Technology
              - Red: Legal Services 
              - Green: Finance
            - First level shows regions, second level shows industries within each region
            - Larger boxes indicate higher total leased space
            """)

with tab2:
    # Input form for recommendations
    st.subheader("Get Personalized City Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        # Region selection
        region = st.selectbox(
            "Select your region of interest:",
            ["Northeast", "Midwest", "South", "West"],
            key="region_select"
        )
        
        # Office size selection
        office_size = st.selectbox(
            "Select office size category:",
            ["< 5,000 SF", "5,000-10,000 SF", "10,000-25,000 SF", "> 25,000 SF"],
            key="size_select"
        )

    with col2:
        # Industry selection - always include the three target industries
        industries = ["Technology", "Legal Services", "Finance"]
        industry = st.selectbox(
            "Select your industry:",
            industries,
            key="industry_select"
        )
        
        # Rent class selection
        rent_class = st.selectbox(
            "Select preferred rent class:",
            ["< $20/SF", "$20-30/SF", "$30-45/SF", "$45-65/SF", "> $65/SF"],
            key="rent_select"
        )

    # Show debug information
    st.write(f"Available regions in model: {list(model.models.keys())}")

    # Button to trigger recommendation
    if st.button("Get Recommendations", type="primary", key="rec_button"):
        with st.spinner("Generating recommendations..."):
            # Debug info
            st.write(f"Selected parameters: Region={region}, Size={office_size}, Industry={industry}, Rent={rent_class}")
            
            # Check if region exists in the model
            if region not in model.models:
                st.error(f"No model available for region: {region}")
                st.info("Please select a different region. Available regions: " + ", ".join(model.models.keys()))
            elif region == "Midwest":
                st.error("No data available for Midwest region.")
                st.info("Please select a different region such as Northeast, South, or West.")
            else:
                # Try to get predictions
                try:
                    prediction = model.predict(region, office_size, industry, rent_class)
                    
                    if prediction:
                        # Display adjustment message if parameters were adjusted
                        if prediction.get('Adjusted Parameters', False):
                            st.warning(f"Note: {prediction['Adjustment Message']} Showing results with adjusted parameters.")
                            
                        # Split into columns
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.success(f"## Recommended City: {prediction['Best City']}")
                            st.info(f"**Probability Score: {prediction['Best City Probability']*100:.2f}%**")
                            
                            # Create metrics
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            
                            with col_metrics1:
                                st.metric("Rent", f"${prediction['Rent']:.2f}/SF")
                            
                            with col_metrics2:
                                st.metric("Crime Rate", f"{prediction['Crime Rate']*100:.2f}%")
                            
                            with col_metrics3:
                                st.metric("Property Tax", f"{prediction['Property Tax']*100:.2f}%")
                            
                            # Display explanation
                            # Use original industry for display if available
                            display_industry = prediction.get('Original Industry', industry)
                            st.markdown(f"""
                            **Why {prediction['Best City']}?**
                            
                            Based on your preferences for a {office_size.lower()} office in the {display_industry} industry 
                            with {rent_class.lower()} rent in the {region} region, {prediction['Best City']} offers 
                            the optimal balance of rent, safety, occupancy, and property tax rates.
                            """)
                            
                            # Show all probabilities in a table
                            st.subheader("All City Probabilities")
                            prob_df = pd.DataFrame({
                                'City': list(prediction['All Probabilities'].keys()),
                                'Probability (%)': [f"{p*100:.2f}%" for p in prediction['All Probabilities'].values()]
                            })
                            st.dataframe(prob_df, hide_index=True)
                        
                        with col2:
                            st.subheader("Alternative Options")
                            
                            # Display alternatives
                            for city, data in prediction["Alternatives"].items():
                                with st.expander(f"{city} (Score: {data['score']*100:.1f}%)"):
                                    st.markdown(f"""
                                    - Rent: ${data['rent']:.2f}/SF
                                    - Crime Rate: {data['crime']*100:.2f}%
                                    - Occupancy: {data['occupancy']*100:.2f}%
                                    - Property Tax: {data['tax']*100:.2f}%
                                    """)
                    else:
                        st.error("Unable to generate recommendations. Please try different criteria.")
                        
                        # Show more detailed error information
                        st.info("""
                        Possible reasons for failure:
                        - The combination of parameters doesn't match any data in our system
                        - There might be missing data for this specific combination
                        
                        Try selecting different values, especially for Office Size and Rent Class.
                        """)
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
                    st.info("Please try a different combination of parameters.")

# Add a note about the data
st.info("""
The recommendations are based on historical commercial real estate data including lease information, 
occupancy rates, regional metrics, and market conditions. The model takes into account office size requirements, 
industry-specific patterns, and rent preferences to suggest optimal locations.
""")

# Footer with additional information
st.markdown("---")
st.markdown("""
**Data Sources:**
- Historical commercial real estate leases
- Market occupancy rates
- Regional crime statistics
- Property tax information
""") 