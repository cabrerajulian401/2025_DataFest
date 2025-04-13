import streamlit as st
import os
import pandas as pd
import plotly.express as px
import sys
import pydeck as pdk
from models import RealEstateModel
import streamlit.components.v1 as components
import time
import traceback
import requests
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print("Loaded environment variables from .env file")
print(f"TAVILY_API_KEY present: {'TAVILY_API_KEY' in os.environ}")
print(f"FETCH_FULL_PAGE: {os.environ.get('FETCH_FULL_PAGE')}")
print(f"MAX_WEB_RESEARCH_LOOPS: {os.environ.get('MAX_WEB_RESEARCH_LOOPS')}")

# Page configuration - MUST be the first st command
st.set_page_config(
    page_title="Commercial Real Estate Recommendation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import LangGraph agent
from ollama_deep_researcher.graph import graph, create_progress_callback, ProgressCallbackHandler, SimpleProgressHandler
from ollama_deep_researcher.state import SummaryStateInput
from ollama_deep_researcher.configuration import Configuration

# Debug - Display session state values to troubleshoot resets
st.write("Debug - Session State:", {k: v for k, v in st.session_state.items() if k not in ['_custom_theme']})

# Initialize session state variables if they don't exist yet
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False
if 'ollama_available' not in st.session_state:
    st.session_state.ollama_available = False
# Initialize additional session state for button callbacks
if 'generate_recs' not in st.session_state:
    st.session_state.generate_recs = False
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "Northeast"
if 'selected_office_size' not in st.session_state:
    st.session_state.selected_office_size = "< 5,000 SF"
if 'selected_industry' not in st.session_state:
    st.session_state.selected_industry = "Technology"
if 'selected_rent_class' not in st.session_state:
    st.session_state.selected_rent_class = "< $20/SF"
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'analysis_stage' not in st.session_state:
    st.session_state.analysis_stage = 0
if 'auto_started' not in st.session_state:
    st.session_state.auto_started = False

# Parse command line arguments for terminal-only mode
parser = argparse.ArgumentParser(description='Commercial Real Estate Recommendation System')
parser.add_argument('--terminal-log', action='store_true', help='Log progress to terminal only')
args, unknown = parser.parse_known_args()

# Store the terminal log preference in an environment variable
if args.terminal_log:
    os.environ["PROGRESS_LOG_TERMINAL"] = "true"
    print("Terminal logging mode enabled. Progress will be displayed in the terminal.")

# Function to check if Ollama is available
def check_ollama_availability(base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            st.session_state.ollama_available = True
            st.session_state.available_models = [model["name"] for model in models]
            return True, models
        return False, []
    except Exception as e:
        st.session_state.ollama_available = False
        return False, []

# Try to check Ollama availability at startup
check_ollama_availability()

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

@st.cache_resource
def load_model():
    """Load and cache the real estate model to prevent retraining on each page refresh"""
    with st.spinner('Loading data and building models...'):
        model = RealEstateModel(data_dir)
        model.load_data()
        model.process_data()
        model.build_models()
    return model

# Define a manual tracking function for node progress
def track_progress_manually(node_name=None, percentage=None, message=None):
    """Update progress indicators manually without relying on callbacks."""
    if 'progress_bar' in st.session_state and 'status_text' in st.session_state and 'message_container' in st.session_state:
        # Use provided values or defaults
        if percentage is not None:
            st.session_state.progress_bar.progress(percentage / 100)
        
        if node_name is not None:
            node_names = {
                "generate_query": "Generating search query",
                "web_research": "Researching market data",
                "summarize_sources": "Analyzing gathered information",
                "reflect_on_summary": "Evaluating analysis completeness",
                "finalize_summary": "Creating final report",
                "END": "Completed analysis"
            }
            readable_name = node_names.get(node_name, node_name)
            st.session_state.status_text.markdown(f"### {readable_name}...")
            
            # Add node-specific messages
            message_html = f"""
            <div style="border-left: 3px solid #1E88E5; padding-left: 15px; margin: 10px 0;">
                <p><i>📊 {readable_name} ({percentage if percentage is not None else 0}% complete)</i></p>
            """
            
            if message:
                message_html += f"<p><i>{message}</i></p>"
            
            message_html += "</div>"
            st.session_state.message_container.markdown(message_html, unsafe_allow_html=True)

class DummyCallback:
    """A dummy callback object that has no methods or attributes.
    
    This is a last resort approach if none of the standard callbacks work.
    """
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        pass

# Function to run the AI analysis
def run_ai_analysis():
    """Run AI analysis when button is clicked, with streaming progress updates"""
    # This now streams real progress as the analysis is executed
    if not st.session_state.ollama_available:
        st.error("⚠️ Ollama service is not available. Cannot generate AI-powered analysis.")
        
        # Offer static analysis option - simplified for the streaming approach
        return generate_static_analysis(get_prediction_data())
        
    # Proceed with Ollama-based analysis if available
    if "model" not in st.session_state or not st.session_state.model:
        # Select best available model
        preferred_models = ["llama3", "llama2", "mistral", "gemma"]
        available_models = st.session_state.get("available_models", [])
        
        selected_model = None
        for model in preferred_models:
            if any(model in m.lower() for m in available_models):
                matching_models = [m for m in available_models if model in m.lower()]
                # Prefer models with higher version numbers or without suffixes
                matching_models.sort(key=lambda x: (len(x.split(":")[0]), x), reverse=True)
                selected_model = matching_models[0]
                break
                
        if not selected_model and available_models:
            # Just pick the first one if none of our preferred models are available
            selected_model = available_models[0]
            
        if not selected_model:
            st.error("No Ollama models are available. Please install at least one model.")
            return None
            
        st.session_state.model = selected_model
    
    try:
        prediction_data = get_prediction_data()
        
        if not prediction_data:
            st.error("No prediction data available")
            return None
        
        # Configure the agent with prediction data
        agent_input = SummaryStateInput(
            research_topic=f"Commercial real estate analysis for {prediction_data['Best City']}",
            recommended_city=prediction_data["Best City"],
            recommended_region=st.session_state.get('region', 'Unknown'),
            industry_type=st.session_state.get('industry', 'Technology'),
            probability_score=prediction_data.get("Best City Probability", 0),
            crime_rate=prediction_data.get("Crime Rate", 0),
            property_tax=prediction_data.get("Property Tax", 0),
            rent_class=st.session_state.get('rent_class', 'Unknown'),
            office_size=st.session_state.get('office_size', 'Unknown'),
            rent_value=prediction_data.get("Rent", 0),
            user_priorities="balanced consideration of rent, safety, and property tax rates"
        )

        # Define the update function to update UI elements
        def update_progress(progress_data):
            """Update Streamlit progress elements with real-time progress data"""
            # Always log to terminal if enabled
            if os.environ.get("PROGRESS_LOG_TERMINAL") == "true":
                percentage = progress_data.get('percentage', 0)
                readable_name = progress_data.get('readable_name', '')
                node = progress_data.get('node', '')
                terminal_msg = f"[PROGRESS] {readable_name} ({percentage}% complete)"
                print("\n" + "="*50)
                print(terminal_msg)
                print("="*50 + "\n")
            
            # Update UI elements if they exist
            if 'progress_bar' in st.session_state and 'status_text' in st.session_state and 'message_container' in st.session_state:
                percentage = progress_data.get('percentage', 0)
                readable_name = progress_data.get('readable_name', '')
                node = progress_data.get('node', '')
                
                # Update the progress bar
                st.session_state.progress_bar.progress(percentage / 100)
                
                # Update the status text
                st.session_state.status_text.markdown(f"### {readable_name}...")
                
                # Add node-specific messages
                message_html = f"""
                <div style="border-left: 3px solid #1E88E5; padding-left: 15px; margin: 10px 0;">
                    <p><i>📊 {readable_name} ({percentage}% complete)</i></p>
                """
                
                if node == "generate_query":
                    message_html += """
                    <p><i>🔍 Constructing search queries based on your requirements...</i></p>
                    """
                elif node == "web_research":
                    message_html += """
                    <p><i>🌐 Gathering market intelligence from reliable sources...</i></p>
                    <p><i>📈 Finding the latest trends and data...</i></p>
                    """
                elif node == "summarize_sources":
                    message_html += """
                    <p><i>📊 Analyzing information relevance to your industry...</i></p>
                    <p><i>🧩 Connecting data points for comprehensive insights...</i></p>
                    """
                elif node == "reflect_on_summary":
                    message_html += """
                    <p><i>🤔 Evaluating analysis completeness and identifying gaps...</i></p>
                    <p><i>🔄 Determining if additional research is needed...</i></p>
                    """
                elif node == "finalize_summary":
                    message_html += """
                    <p><i>📝 Formatting report with executive summary and key metrics...</i></p>
                    <p><i>🎯 Finalizing actionable recommendations...</i></p>
                    """
                
                message_html += "</div>"
                st.session_state.message_container.markdown(message_html, unsafe_allow_html=True)
                
        # Try using each approach with proper error handling for callback issues
        try:
            # Set initial progress manually
            track_progress_manually("generate_query", 10, "Starting the analysis process...")
            
            # APPROACH 1: Try with simplified callback
            try:
                print("Attempting analysis with simplified callback...")
                progress_callback = create_progress_callback(update_progress, simplified=True)
                
                # Update progress before starting
                track_progress_manually("generate_query", 20, "Generating search query...")
                
                # First, try with the callback
                result = graph.invoke(
                    agent_input,
                    {"callbacks": [progress_callback]}
                )
                
                # If we get here, the process completed successfully
                track_progress_manually("END", 100, "Analysis completed successfully!")
                return result
                
            except AttributeError as attr_error:
                # This is likely a callback compatibility issue
                print(f"Callback error: {str(attr_error)}")
                print("Retrying with standard callback...")
                
                # APPROACH 2: Try with standard callback
                try:
                    track_progress_manually("generate_query", 20, "Trying standard callback approach...")
                    progress_callback = create_progress_callback(update_progress, simplified=False)
                    
                    result = graph.invoke(
                        agent_input,
                        {"callbacks": [progress_callback]}
                    )
                    
                    track_progress_manually("END", 100, "Analysis completed successfully!")
                    return result
                    
                except Exception as callback_error:
                    print(f"Standard callback error: {str(callback_error)}")
                    print("Trying with minimal callback...")
                    
                    # Try with minimal callback
                    try:
                        track_progress_manually("generate_query", 20, "Using minimal callback...")
                        progress_callback = create_progress_callback(update_progress, minimal=True)
                        
                        result = graph.invoke(
                            agent_input,
                            {"callbacks": [progress_callback]}
                        )
                        
                        track_progress_manually("END", 100, "Analysis completed successfully!")
                        return result
                    except Exception as minimal_error:
                        print(f"Minimal callback error: {str(minimal_error)}")
                        print("Trying with dummy callback...")
                        
                        # Try with dummy callback
                        try:
                            track_progress_manually("generate_query", 20, "Using dummy callback...")
                            dummy_callback = DummyCallback()
                            
                            result = graph.invoke(
                                agent_input,
                                {"callbacks": [dummy_callback]}
                            )
                            
                            track_progress_manually("END", 100, "Analysis completed successfully!")
                            return result
                        except Exception as dummy_error:
                            print(f"Dummy callback error: {str(dummy_error)}")
                            print("Retrying without any callbacks...")
                            
                            # APPROACH 4: No callbacks but manual tracking
                            try:
                                # Reset progress for clarity
                                track_progress_manually("generate_query", 20, "Running without callbacks...")
                                
                                # Create a list of stages we'll manually track
                                stages = [
                                    ("generate_query", 20, "Generating search query..."),
                                    ("web_research", 40, "Researching market data..."),
                                    ("summarize_sources", 60, "Analyzing research data..."),
                                    ("reflect_on_summary", 80, "Evaluating completeness..."),
                                    ("finalize_summary", 95, "Creating final report...")
                                ]
                                
                                # Display first stage
                                track_progress_manually(*stages[0])
                                
                                # Basic invoke without callbacks
                                result = graph.invoke(agent_input)
                                
                                # Simulate completion through all stages
                                for stage in stages[1:]:
                                    time.sleep(0.5)  # Small delay to show progression
                                    track_progress_manually(*stage)
                                
                                # Mark as complete
                                track_progress_manually("END", 100, "Analysis completed successfully!")
                                return result
                            
                            except Exception as no_callback_error:
                                print(f"Error without callbacks: {str(no_callback_error)}")
                                # Fall back to static analysis as a last resort
                                print("Using static analysis as final fallback")
                                track_progress_manually("END", 100, "Using static analysis fallback...")
                                return generate_static_analysis(prediction_data)
            
            except Exception as general_error:
                print(f"Unexpected error during analysis: {str(general_error)}")
                print("Trying without callbacks...")
                
                # Try without callbacks
                track_progress_manually("generate_query", 20, "Running without callback tracking...")
                result = graph.invoke(agent_input)
                track_progress_manually("END", 100, "Analysis completed successfully!")
                return result
                
        except Exception as e:
            print(f"Error in AI analysis function: {str(e)}")
            # Set error state in the progress
            if 'status_text' in st.session_state:
                st.session_state.status_text.markdown("### ❌ Analysis Failed")
            
            # Final fallback - static analysis
            print("All approaches failed. Using static analysis...")
            track_progress_manually("END", 100, "Using static analysis fallback...")
            return generate_static_analysis(prediction_data)
        
    except Exception as e:
        st.error(f"Error in AI analysis function: {str(e)}")
        # Last resort fallback
        try:
            return generate_static_analysis(prediction_data)
        except:
            st.error("Cannot generate any analysis. Please try again later.")
            return None

def generate_static_analysis(prediction):
    """
    Generate a static market analysis without using Ollama LLM.
    This is a fallback option when the LLM service is not available.
    
    Args:
        prediction: Dictionary containing prediction data
        
    Returns:
        A SimpleNamespace object with running_summary attribute containing the static analysis
    """
    from types import SimpleNamespace
    
    if prediction is None:
        st.error("Cannot generate analysis without prediction data")
        return None
    
    # Get basic information
    city = prediction["Best City"]
    rent = prediction.get("Rent", "N/A")
    crime_rate = prediction.get("Crime Rate", 0) * 100
    property_tax = prediction.get("Property Tax", 0) * 100
    probability = prediction.get("Best City Probability", 0) * 100
    
    # Get parameters from session state
    region = st.session_state.get('region', 'Unknown')
    industry = st.session_state.get('industry', 'Unknown')
    rent_class = st.session_state.get('rent_class', 'Unknown')
    office_size = st.session_state.get('office_size', 'Unknown')
    
    # Generate alternative cities list
    alternatives = ", ".join([city for city in prediction["Alternatives"].keys()]) if "Alternatives" in prediction else ""
    
    # Create a well-formatted commercial real estate report
    report_header = f"""# Commercial Real Estate Analysis: {city}, {region}

## Executive Summary
**Industry Focus:** {industry}
**Office Size Requirements:** {office_size}
**Rent Class:** {rent_class}
**Rent Value:** ${rent}/SF
**Recommendation Confidence:** {probability:.1f}%
**Alternative Locations Considered:** {alternatives}
**User Priorities:** Optimizing for rent, safety, and property tax rates

### Key Metrics
| Metric | Value | Impact |
|--------|-------|--------|
| **Rent** | ${rent}/SF | {'High' if float(rent) > 40 else 'Moderate' if float(rent) > 25 else 'Low'} cost impact |
| **Crime Rate** | {crime_rate:.1f}% | {'High' if crime_rate > 8 else 'Moderate' if crime_rate > 5 else 'Low'} security concern |
| **Property Tax** | {property_tax:.1f}% | {'High' if property_tax > 3 else 'Moderate' if property_tax > 2 else 'Low'} tax burden |
"""

    # Add comparison table
    comparison_table = "### City Comparison\n"
    comparison_table += "| City | Probability | Rent | Crime Rate | Property Tax | Overall Rating |\n"
    comparison_table += "|------|-------------|------|------------|-------------|---------------|\n"
    
    # Add primary city row
    primary_score = min(max(round(5 - (float(rent)/25 + crime_rate/5 + property_tax/2)/3), 1), 5) if rent != "N/A" else 3
    primary_row = f"| **{city}** | {probability:.1f}% | "
    primary_row += f"${rent}/SF | "
    primary_row += f"{crime_rate:.1f}% | "
    primary_row += f"{property_tax:.1f}% | "
    primary_row += f"{primary_score} {'★' * primary_score}{'☆' * (5-primary_score)} |\n"
    comparison_table += primary_row
    
    # Add alternative cities
    if "Alternatives" in prediction:
        for alt_city, data in prediction["Alternatives"].items():
            alt_rent = data.get('rent', 0)
            alt_crime = data.get('crime', 0) * 100
            alt_tax = data.get('tax', 0) * 100
            alt_score = min(max(round(5 - (float(alt_rent)/25 + alt_crime/5 + alt_tax/2)/3), 1), 5)
            
            alt_row = f"| {alt_city} | {data.get('score', 0) * 100:.1f}% | "
            alt_row += f"${alt_rent}/SF | "
            alt_row += f"{alt_crime:.1f}% | "
            alt_row += f"{alt_tax:.1f}% | "
            alt_row += f"{alt_score} {'★' * alt_score}{'☆' * (5-alt_score)} |\n"
            comparison_table += alt_row
    
    # Add highlights section
    highlights = f"""
## Recommendation Highlights

### Why {city}?
- **Industry Alignment:** Selected for optimal conditions for {industry} businesses
- **Cost Efficiency:** Balances competitive rent with operational advantages
- **Strategic Location:** Positioned for {industry} market access and growth
- **Risk Factors:** Assessed and determined to be manageable given the business requirements

### Key Decision Factors
- Confidence score of {probability:.1f}% based on comprehensive data analysis
- Prioritized rent, safety, and property tax rates in the selection process
- Compared against alternatives: {alternatives}
- Considered specific requirements for {office_size} office space
"""

    # Add static market analysis based on industry
    market_analysis = """
## Detailed Analysis

### Market Overview
This market analysis is based on static data and does not include real-time web research. For a comprehensive analysis with the latest market data, please ensure Ollama LLM service is running correctly and try again.

The commercial real estate market in this region has shown stabilization in recent quarters after periods of fluctuation. Vacancy rates remain within acceptable ranges for the selected market, and lease terms are generally negotiable for tenants with strong credit profiles.

### Industry-Specific Considerations
"""

    if industry == "Technology":
        market_analysis += """
Technology companies often prioritize locations with:
- Access to talent pools and educational institutions
- Proximity to innovation hubs and incubators
- Modern infrastructure with high-speed connectivity
- Flexible spaces that can accommodate growth
- Amenities that attract and retain tech talent

The recommended location provides a balanced approach to these considerations while maintaining reasonable occupancy costs.
"""
    elif industry == "Legal Services":
        market_analysis += """
Legal service firms typically require:
- Proximity to courthouses and government offices
- Professional environments that convey stability and prestige
- Client-accessible locations with good transportation links
- Secure facilities for confidential information
- Traditional office layouts with appropriate meeting spaces

The selected location offers the professional environment needed while balancing operational costs effectively.
"""
    elif industry == "Finance":
        market_analysis += """
Financial service operations generally seek:
- Locations near other financial institutions
- Secure and professional environments
- Prestigious addresses that instill client confidence
- Good transportation access for clients and employees
- Spaces that provide appropriate security measures

The recommended market provides these attributes while maintaining reasonable occupancy costs compared to alternatives.
"""
    else:
        market_analysis += f"""
{industry} businesses typically require locations that balance cost considerations with industry-specific needs. The recommended market provides appropriate infrastructure, accessibility, and professional environment while maintaining reasonable occupancy costs.
"""

    # Add risk assessment
    risk_assessment = """
### Risk Assessment

**Economic Factors**
- General economic trends remain positive though with potential headwinds from broader market conditions
- Local business environment supports continued demand for commercial space
- Development pipeline is balanced with absorption rates

**Market-Specific Considerations**
- Rental rates have shown moderate growth and are projected to remain stable
- Vacancy rates are within sustainable ranges
- New development is occurring at a measured pace

**Potential Challenges**
- Changes in work patterns may impact future space requirements
- Economic uncertainty could affect expansion decisions
- Competitive markets may offer incentives to attract tenants

**Mitigation Strategies**
- Negotiate flexibility in lease terms where possible
- Consider phased occupancy strategies
- Evaluate subleasing options if requirements change
"""

    # Combine all sections
    static_analysis = f"{report_header}\n{comparison_table}\n{highlights}\n{market_analysis}\n{risk_assessment}\n\n## Sources\nThis is a static analysis based on available data. No real-time web research was performed."
    
    # Create a simple object with running_summary attribute to mimic the LangGraph output
    result = SimpleNamespace(running_summary=static_analysis)
    return result

def get_prediction_data():
    """Get prediction data from session state or return an error message"""
    if st.session_state.get('prediction_data') is None:
        st.error("No prediction data available. Please generate recommendations first.")
        return None
    
    return st.session_state.prediction_data

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
            - Compare markets across multiple metrics at once:
              - Rent ($/SF)
              - Crime Rate (%)
              - Property Tax (%)
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
    # Show previously generated analysis if it exists
    if (st.session_state.get('analysis_result') is not None and 
        hasattr(st.session_state.analysis_result, 'running_summary') and
        not st.session_state.get('analysis_in_progress', False)):
        
        st.info("Showing previously generated analysis. You can generate a new one by selecting parameters and clicking the recommendation button below.")
        
        # Display the report with an eye-catching header
        st.markdown("""
        <style>
        .report-header {
            background-color: #1E88E5;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        </style>
        <div class="report-header">
            <h2>📈 Executive Market Intelligence Report</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the saved analysis
        st.markdown(st.session_state.analysis_result)
        
        # Add a button to clear the previous analysis
        if st.button("🗑️ Clear Previous Analysis", key="clear_analysis"):
            st.session_state.analysis_result = None
            st.experimental_rerun()
        
        st.markdown("---")
    
    # Input form for recommendations
    st.subheader("Get Personalized City Recommendations")

    # Use a form to batch all input changes and reduce reruns
    with st.form(key="recommendation_form"):
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

        # Show available regions
        st.write(f"Available regions in model: {list(model.models.keys())}")

        # Submit button for the form
        submitted = st.form_submit_button("Get Recommendations", type="primary")
        
        if submitted:
            # Store parameters in session state to preserve them
            st.session_state.selected_region = region
            st.session_state.selected_office_size = office_size
            st.session_state.selected_industry = industry 
            st.session_state.selected_rent_class = rent_class
            st.session_state.generate_recs = True
    
    # Check if we should generate recommendations (outside the form)
    if st.session_state.get('generate_recs', False):
        with st.spinner("Generating recommendations..."):
            # Get values from session state
            region = st.session_state.selected_region
            office_size = st.session_state.selected_office_size
            industry = st.session_state.selected_industry
            rent_class = st.session_state.selected_rent_class
            
            # Reset the flag
            st.session_state.generate_recs = False
            
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
                            the optimal balance of rent, safety, and property tax rates.
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
                        
                    # If we have a valid prediction, add the AI Market Analysis button
                    if prediction and 'Best City' in prediction:
                        st.markdown("---")
                        
                        # More prominent styling for the AI analysis section
                        st.markdown("""
                        <style>
                        .ai-analysis-header {
                            font-size: 32px;
                            color: #1E88E5;
                            margin-bottom: 20px;
                        }
                        .ai-analysis-container {
                            background-color: #f8f9fa;
                            border-radius: 10px;
                            padding: 20px;
                            border-left: 5px solid #1E88E5;
                            margin-bottom: 20px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Larger, more prominent header
                        st.markdown("<div class='ai-analysis-header'>🔍 AI-Powered Market Analysis</div>", unsafe_allow_html=True)
                        
                        # Enhanced container for analysis description
                        st.markdown("<div class='ai-analysis-container'>", unsafe_allow_html=True)
                        st.markdown("""
                        ### Transform your decision-making with advanced AI research
                        
                        Our AI-powered analysis engine delivers comprehensive, data-driven insights by analyzing thousands of market data points and recent commercial real estate trends. Get a detailed report including:
                        
                        ✅ **Current Market Conditions**: Up-to-date commercial real estate trends and pricing dynamics
                        
                        ✅ **Economic Environment**: Key economic indicators and business climate analysis
                        
                        ✅ **Industry-Specific Insights**: Tailored considerations for your specific industry needs
                        
                        ✅ **Risk Assessment**: Identification of potential risk factors and future market outlook
                        
                        ✅ **Competitive Comparison**: Side-by-side analysis with alternative locations
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Check Ollama status and display it
                        ollama_status_container = st.container()
                        with ollama_status_container:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if st.session_state.ollama_available:
                                    st.success(f"✓ Ollama service is available with {len(st.session_state.get('available_models', []))} model(s)")
                                    if st.session_state.get('available_models'):
                                        st.info(f"Available models: {', '.join(st.session_state.get('available_models', []))}")
                                else:
                                    st.error("⚠️ Ollama service is not available. Analysis might not work.")
                                    st.info("Please ensure Ollama is running on your machine and models are downloaded.")
                            
                            with col2:
                                if st.button("Check Again", key="check_ollama"):
                                    available, models = check_ollama_availability()
                                    if available:
                                        st.session_state.ollama_availability_message = f"Connected to Ollama with {len(models)} models"
                                    else:
                                        st.session_state.ollama_availability_message = "Could not connect to Ollama"
                        
                        # Start analysis automatically if Ollama is available
                        if st.session_state.ollama_available and not st.session_state.get('analysis_started', False) and not st.session_state.get('analysis_in_progress', False):
                            # Save important parameters to session state
                            st.session_state.region = region
                            st.session_state.industry = industry
                            st.session_state.rent_class = rent_class
                            st.session_state.office_size = office_size
                            st.session_state.prediction_data = prediction
                            st.session_state.analysis_started = True
                            st.session_state.analysis_in_progress = True
                            st.session_state.analysis_stage = 0
                            st.session_state.auto_started = True
                            # Will pick up in next rerun
                            
                        # Provide option to manually start analysis if auto-start is disabled or failed
                        else:
                            button_disabled = not st.session_state.ollama_available
                            
                            if button_disabled:
                                st.warning("Analysis button disabled: Ollama service is required to generate analysis.")
                            
                            # Define callback function that will be called when button is clicked
                            def start_analysis_workflow():
                                # Save important parameters to session state
                                st.session_state.region = region
                                st.session_state.industry = industry
                                st.session_state.rent_class = rent_class
                                st.session_state.office_size = office_size
                                st.session_state.prediction_data = prediction
                                st.session_state.analysis_started = True
                                st.session_state.analysis_in_progress = True
                                st.session_state.analysis_stage = 0
                                
                            # Use on_click parameter to prevent page reset
                            st.button(
                                "📊 GENERATE COMPREHENSIVE MARKET ANALYSIS", 
                                type="primary", 
                                key="ai_analysis_button", 
                                on_click=start_analysis_workflow,
                                use_container_width=True,
                                disabled=button_disabled
                            )
                        
                        # Status message for automatic analysis
                        if st.session_state.get('auto_started', False) and not st.session_state.get('analysis_started', False):
                            st.success("AI analysis completed automatically!")
                            
                        # Handle the analysis workflow if it has started
                        if st.session_state.get('analysis_started', False):
                            # Create persistent containers for progress display that will survive reruns
                            if 'progress_bar' not in st.session_state:
                                progress_container = st.container()
                                with progress_container:
                                    st.session_state.progress_bar = st.progress(0)
                                    st.session_state.status_text = st.empty()
                                    st.session_state.message_container = st.empty()
                            
                            # Get the persistent containers from session state
                            progress_bar = st.session_state.progress_bar
                            status_text = st.session_state.status_text
                            message_container = st.session_state.message_container
                            
                            try:
                                # If we haven't started the actual analysis yet, initialize it
                                if 'analysis_executing' not in st.session_state or not st.session_state.analysis_executing:
                                    # Set initial progress state
                                    progress_bar.progress(0)
                                    status_text.markdown("### Starting Analysis...")
                                    message_container.markdown("""
                                    <div style="border-left: 3px solid #1E88E5; padding-left: 15px; margin: 10px 0;">
                                        <p><i>🚀 Initializing market analysis engine...</i></p>
                                        <p><i>⚙️ Configuring parameters for your specific requirements...</i></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Mark that we're executing so we don't initialize again
                                    st.session_state.analysis_executing = True
                                    
                                    # Run the actual analysis - this will update progress via callback
                                    result = run_ai_analysis()
                                    st.session_state.analysis_result = result
                                    
                                    # Analysis is complete
                                    st.session_state.analysis_executing = False
                                    st.session_state.analysis_in_progress = False
                                    st.session_state.analysis_started = False
                                    
                                    # Display success message
                                    st.success("Analysis completed successfully!")
                                    
                                    # Display the report with an eye-catching header
                                    st.markdown("""
                                    <style>
                                    .report-header {
                                        background-color: #1E88E5;
                                        color: white;
                                        padding: 10px 15px;
                                        border-radius: 5px;
                                        margin-bottom: 20px;
                                    }
                                    </style>
                                    <div class="report-header">
                                        <h2>📈 Executive Market Intelligence Report</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display the report if available
                                    if result and hasattr(result, 'running_summary'):
                                        st.markdown(result.running_summary)
                                    else:
                                        st.error("Failed to generate analysis. Please check system logs for details.")
                                else:
                                    # If already executing, just show a waiting message
                                    st.info("Analysis in progress... Please wait while we generate your report.")
                            
                            except Exception as e:
                                st.error(f"Error generating market analysis: {str(e)}")
                                st.info("Our research system is temporarily unavailable. Please try again later.")
                                st.session_state.analysis_in_progress = False
                                st.session_state.analysis_started = False
                                st.session_state.analysis_executing = False
                                # Display the full error details for debugging
                                st.error(traceback.format_exc())
                                
                                # Add a more user-friendly error message and retry option
                                error_container = st.container()
                                with error_container:
                                    st.markdown("""
                                    <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; border-left: 5px solid #dc3545;">
                                        <h3 style="color: #dc3545;">Analysis Generation Error</h3>
                                        <p>We encountered an issue while generating your market analysis. This could be due to:</p>
                                        <ul>
                                            <li>Temporary connectivity issues with our research engine</li>
                                            <li>High system demand at the moment</li>
                                            <li>Insufficient data for the specific location/industry combination</li>
                                        </ul>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Define retry callback
                                    def retry_analysis():
                                        st.session_state.analysis_started = False
                                        st.session_state.analysis_executing = False
                                    
                                    # Define clear error callback
                                    def clear_error_state():
                                        st.session_state.analysis_in_progress = False
                                        st.session_state.analysis_error = None
                                        st.session_state.analysis_started = False
                                        st.session_state.analysis_executing = False
                                    
                                    # Add retry options with callbacks
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.button("🔄 Retry Analysis", 
                                                 key="retry_analysis",
                                                 on_click=retry_analysis)
                                    with col2:
                                        st.button("🧹 Clear Error State", 
                                                 key="clear_error",
                                                 on_click=clear_error_state)
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