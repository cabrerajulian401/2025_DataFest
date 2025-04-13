# Commercial Real Estate Recommendation System

This Streamlit application helps businesses find the optimal location for their commercial real estate needs based on industry, region preferences, price range, and office size requirements.

## Features

- Industry-based location recommendations
- Multi-region analysis
- Price range and office size filtering
- Key metrics for each recommendation:
  - Average Rent ($/SF)
  - Crime Rate
  - Occupancy Rate
  - Property Tax
- Alternative city recommendations with detailed metrics

## New Feature: Real-time Analysis Progress Streaming

The application now provides real-time progress updates during the AI-powered market analysis process. This allows users to see exactly what's happening at each step of the analysis.

### Features:

- **Visual Progress Bar**: Shows the current completion percentage
- **Status Messages**: Displays the current stage of analysis with descriptive text
- **Detailed Updates**: Provides specific information about what's being processed at each stage
- **Terminal Logging Option**: Ability to track progress in the terminal for headless environments

### How It Works:

1. The application uses LangGraph's callback system to track progress through each node of the workflow
2. Each node in the graph (query generation, web research, summarizing, etc.) reports its progress
3. Progress is streamed to both the UI and optionally to the terminal
4. The progress bar and messages update in real-time as the analysis moves through different stages

### Usage:

To run the application with UI progress visualization (default):
```bash
streamlit run app/realestate_app.py
```

To run with terminal-only progress logging (useful for debugging or headless environments):
```bash
streamlit run app/realestate_app.py --terminal-log
```

## Technical Implementation:

- Added a callback mechanism in `graph.py` to track node transitions
- Each node reports its progress percentage and current activity
- Progress can be displayed in the UI, terminal, or logged to a file
- The LangGraph workflow is now completely transparent to users

## Requirements:

- Python 3.7+
- Streamlit
- LangGraph
- Ollama (for LLM inference)
- Other dependencies in requirements.txt

## Installation

1. Clone this repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Setting Up R Integration (Optional)

The app can run with mock data without R, but for real predictions using the models, you need to set up R:

1. **Install R**:
   - Download R from [CRAN](https://cran.r-project.org/bin/windows/base/)
   - Install with default settings
   
2. **Install required R packages**:
   Open R or RStudio and run:
   ```r
   install.packages(c("dplyr", "tidyr", "ggplot2", "plotly", "randomForest", "nnet"))
   ```

3. **Set R_HOME environment variable**:
   - Find your R installation path (usually `C:\Program Files\R\R-x.x.x`)
   - Set it as an environment variable:
     - Windows: `set R_HOME=C:\Program Files\R\R-x.x.x`
     - Linux/Mac: `export R_HOME=/path/to/R`

4. **Verify R installation**:
   Open Command Prompt/Terminal and run:
   ```
   R --version
   ```

## Data Sources

The models use data from the Savills_Data directory:
- Commercial real estate metrics
- Regional pricing information
- Occupancy rates
- Crime statistics
- Property tax information

## Troubleshooting

If you encounter R integration issues:
1. Confirm R is installed and accessible from command line
2. Check that all required R packages are installed
3. Verify R_HOME environment variable is set correctly
4. The app will fall back to mock data if R integration fails

