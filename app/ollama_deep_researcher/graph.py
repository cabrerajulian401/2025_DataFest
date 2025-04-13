import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Callable

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, strip_thinking_tokens, get_config_value
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date
from langchain_core.callbacks.base import BaseCallbackHandler

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the commercial real estate context."""

    # Format the prompt with real estate specific information
    current_date = get_current_date()
    
    # Format crime rate, property tax and probability score as percentages if they exist
    crime_rate = state.crime_rate * 100 if state.crime_rate is not None else "N/A"
    property_tax = state.property_tax * 100 if state.property_tax is not None else "N/A" 
    probability_score = state.probability_score * 100 if state.probability_score is not None else "N/A"
    
    # Get rent value
    rent_value = state.rent_value if hasattr(state, 'rent_value') and state.rent_value else "N/A"
    
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        city=state.recommended_city,
        region=state.recommended_region,
        industry=state.industry_type,
        office_size=state.office_size,
        rent_class=state.rent_class,
        crime_rate=crime_rate,
        property_tax=property_tax,
        probability_score=probability_score,
        rent_value=rent_value
    )

    # If no specific research topic was provided, create one based on the city and industry
    if not state.research_topic:
        state.research_topic = f"Commercial real estate analysis for {state.industry_type} company in {state.recommended_city}, {state.recommended_region}"

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Ensure base_url is properly sanitized
    base_url = configurable.ollama_base_url.strip()
    print(f"Using Ollama base URL: {base_url}")
    
    llm_json_mode = ChatOllama(
        base_url=base_url, 
        model=configurable.local_llm, 
        temperature=0, 
        format="json"
    )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for researching commercial real estate in {state.recommended_city} for {state.industry_type} industry:")]
    )
    
    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        query = json.loads(content)
        search_query = query['query']
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = f"{state.recommended_city} commercial real estate {state.industry_type} {current_date.split()[2]}"
    return {"search_query": search_query}

def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web search using commercial real estate specific queries."""
    
    configurable = Configuration.from_runnable_config(config)
    
    # Debug logging
    print(f"STARTING WEB RESEARCH: loop {state.research_loop_count + 1}")
    print(f"SEARCH QUERY: {state.search_query}")
    print(f"SEARCH API: {configurable.search_api}")
    print(f"FETCH FULL PAGE: {configurable.fetch_full_page}")
    
    try:
        # Use the selected search API
        if configurable.search_api == "tavily":
            from ollama_deep_researcher.utils import tavily_search
            search_results = tavily_search(
                state.search_query,
                fetch_full_page=configurable.fetch_full_page,
                api_key=configurable.tavily_api_key
            )
            print(f"TAVILY SEARCH RESULTS: {len(search_results.get('results', []))} results found")
        else:  # perplexity
            from ollama_deep_researcher.utils import perplexity_search
            search_results = perplexity_search(
                state.search_query,
                perplexity_search_loop_count=state.research_loop_count
            )
            print(f"PERPLEXITY SEARCH RESULTS: {len(search_results.get('results', []))} results found")
        
        from ollama_deep_researcher.utils import deduplicate_and_format_sources
        sources = deduplicate_and_format_sources(
            search_results, 
            max_tokens_per_source=500,
            fetch_full_page=configurable.fetch_full_page
        )
        
        # Increment the research loop counter
        state.web_research_results.append(sources)
        state.sources_gathered.append(sources)
        state.research_loop_count += 1
        
        print(f"WEB RESEARCH COMPLETE: loop {state.research_loop_count}")
        return {"web_research_results": state.web_research_results, "research_loop_count": state.research_loop_count}
    
    except Exception as e:
        print(f"ERROR IN WEB RESEARCH: {str(e)}")
        # Return empty results but still increment counter to avoid getting stuck
        state.research_loop_count += 1
        return {"web_research_results": state.web_research_results, "research_loop_count": state.research_loop_count}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes commercial real estate research findings."""
    
    configurable = Configuration.from_runnable_config(config)
    
    rent_value = state.rent_value if hasattr(state, 'rent_value') and state.rent_value else "N/A"
    
    # Format user priorities
    user_priorities = state.user_priorities if hasattr(state, 'user_priorities') and state.user_priorities else "balanced consideration of all factors"
    
    # Check for most recent web research and existing summaries
    most_recent_web_research = state.web_research_results[-1] if state.web_research_results else ""
    existing_summary = state.running_summary if state.running_summary else ""
    
    # Get industry-specific prompting based on the industry type
    industry_specific_prompt = get_industry_specific_prompt(state.industry_type)
    
    # Create human message content including user priorities and alternative city data
    human_message_content = (
        f"You are generating a professional commercial real estate market analysis for a {state.industry_type} company "
        f"considering {state.office_size} office space in {state.recommended_city} with a budget of {state.rent_class}.\n\n"
        f"The client's key priorities are: {user_priorities}\n\n"
    )
    
    # Add existing summary if available
    if existing_summary:
        human_message_content += (
            f"<Previous Analysis>\n{existing_summary}\n</Previous Analysis>\n\n"
            f"<New Research>\n{most_recent_web_research}\n</New Research>\n\n"
            f"Update the previous analysis with this new research information, maintaining the professional tone and detailed insights. "
            f"Enhance any sections where the new information provides greater clarity or more current data.\n\n"
        )
    else:
        human_message_content += (
            f"<Research Information>\n{most_recent_web_research}\n</Research Information>\n\n"
            f"Create a comprehensive commercial real estate analysis for {state.recommended_city} focusing on the {state.industry_type} industry.\n\n"
        )
    
    # Add industry-specific considerations
    human_message_content += f"{industry_specific_prompt}\n\n"
    
    # Add specific guidance on analytical depth
    human_message_content += (
        f"This report will be used for executive decision-making, so incorporate:\n"
        f"1. Data-driven insights with specific metrics where available\n"
        f"2. Forward-looking market trends and projections\n"
        f"3. Competitive positioning analysis for {state.industry_type} businesses in {state.recommended_city}\n"
        f"4. Risk-adjusted assessment of the recommendation\n"
        f"5. Actionable recommendations with supporting rationale\n"
    )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    
    # Ensure base_url is properly sanitized
    base_url = configurable.ollama_base_url.strip()
    print(f"Using Ollama base URL: {base_url}")
    
    llm = ChatOllama(
        base_url=base_url, 
        model=configurable.local_llm, 
        temperature=0
    )
    
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions.format(
            city=state.recommended_city, 
            industry=state.industry_type,
            rent_value=rent_value,
            crime_rate=state.crime_rate * 100 if state.crime_rate else 'N/A',
            property_tax=state.property_tax * 100 if state.property_tax else 'N/A',
            probability_score=state.probability_score * 100 if state.probability_score else 'N/A',
            user_priorities=user_priorities
        )),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    return {"running_summary": running_summary}

def get_industry_specific_prompt(industry: str) -> str:
    """Return industry-specific research prompts based on the industry type."""
    
    industry_prompts = {
        "Technology": """For Technology companies, please focus on:
- Tech talent availability and educational institutions in the region
- Presence of tech hubs, incubators, or innovation districts
- Digital infrastructure quality (fiber optic networks, data centers, etc.)
- Proximity to potential clients and partners in the tech ecosystem
- Tax incentives specific to technology companies
- Recent tech company relocations or expansions in the area
- Competitive analysis with other tech-focused cities""",
        
        "Legal Services": """For Legal Services firms, please focus on:
- Proximity to courts, government offices, and regulatory agencies
- Concentration of complementary professional services
- Client base analysis and industry concentrations in the area
- Prestige factor of locations and buildings for client-facing operations
- Transportation accessibility for clients and staff
- Historical preference patterns of comparable law firms
- Regulatory environment specific to legal practice""",
        
        "Finance": """For Financial Services companies, please focus on:
- Proximity to other financial institutions and services
- Regulatory environment and compliance considerations
- Security infrastructure and building requirements
- Transportation accessibility for clients and employees
- Prestige factor of locations for client confidence
- Financial industry clustering and ecosystem analysis
- Economic stability indicators specific to the region"""
    }
    
    # Return industry-specific prompt if available, otherwise a generic one
    return industry_prompts.get(industry, f"For {industry} companies, please consider industry-specific factors such as proximity to clients, suppliers, talent pool, regulatory environment, and competitive positioning in the market.")

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps in commercial real estate analysis."""

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Format the reflection instructions with real estate context
    rent_value = state.rent_value if hasattr(state, 'rent_value') and state.rent_value else "N/A"
    user_priorities = state.user_priorities if hasattr(state, 'user_priorities') and state.user_priorities else "balanced consideration of all factors"
    
    formatted_instructions = reflection_instructions.format(
        city=state.recommended_city,
        industry=state.industry_type,
        office_size=state.office_size,
        rent_class=state.rent_class,
        rent_value=rent_value,
        crime_rate=state.crime_rate * 100 if state.crime_rate else 'N/A',
        property_tax=state.property_tax * 100 if state.property_tax else 'N/A',
        user_priorities=user_priorities
    )
    
    # Ensure base_url is properly sanitized
    base_url = configurable.ollama_base_url.strip()
    print(f"Using Ollama base URL: {base_url}")
    
    llm_json_mode = ChatOllama(
        base_url=base_url, 
        model=configurable.local_llm, 
        temperature=0, 
        format="json"
    )

    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_instructions),
        HumanMessage(content=f"Reflect on our existing commercial real estate analysis: \n === \n {state.running_summary}, \n === \n Identify a knowledge gap and generate a follow-up web search query focusing on {state.recommended_city}, prioritizing {user_priorities}:")]
    )
    
    # Parse JSON response and extract follow-up query
    try:
        reflection_content = json.loads(result.content)
        query = reflection_content.get('follow_up_query')
        if not query:
            return {"search_query": f"Recent commercial real estate trends in {state.recommended_city} for {state.industry_type} industry focusing on {user_priorities}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails, use a fallback query
        return {"search_query": f"Recent commercial real estate trends in {state.recommended_city} for {state.industry_type} industry focusing on {user_priorities}"}

def finalize_summary(state: SummaryState):
    """LangGraph node that finalizes the commercial real estate research report."""

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []
    
    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)
    
    # Format for display
    rent_value = state.rent_value if hasattr(state, 'rent_value') and state.rent_value else "N/A"
    user_priorities = state.user_priorities if hasattr(state, 'user_priorities') and state.user_priorities else "balanced consideration of all factors"
    
    # Create a more visually appealing, well-formatted commercial real estate report
    report_header = f"""# Commercial Real Estate Analysis: {state.recommended_city}, {state.recommended_region}

## Executive Summary
**Industry Focus:** {state.industry_type}
**Office Size Requirements:** {state.office_size}
**Rent Class:** {state.rent_class}
**Rent Value:** ${rent_value}/SF
**Recommendation Confidence:** {state.probability_score * 100 if state.probability_score else 'N/A'}%
**User Priorities:** {user_priorities}

### Key Metrics
| Metric | Value | Impact |
|--------|-------|--------|
| **Rent** | ${rent_value}/SF | {'High' if float(rent_value if rent_value != 'N/A' else 0) > 40 else 'Moderate' if float(rent_value if rent_value != 'N/A' else 0) > 25 else 'Low'} cost impact |
| **Crime Rate** | {state.crime_rate * 100 if state.crime_rate else 'N/A'}% | {'High' if state.crime_rate and state.crime_rate > 0.08 else 'Moderate' if state.crime_rate and state.crime_rate > 0.05 else 'Low'} security concern |
| **Property Tax** | {state.property_tax * 100 if state.property_tax else 'N/A'}% | {'High' if state.property_tax and state.property_tax > 0.03 else 'Moderate' if state.property_tax and state.property_tax > 0.02 else 'Low'} tax burden |

"""

    # Add recommendation highlights section
    report_header += f"""
## Recommendation Highlights

### Why {state.recommended_city}?
- **Industry Alignment:** Selected for optimal conditions for {state.industry_type} businesses
- **Cost Efficiency:** Balances competitive rent with operational advantages
- **Strategic Location:** Positioned for {state.industry_type} market access and growth
- **Risk Factors:** Assessed and determined to be manageable given the business requirements

### Key Decision Factors
- Confidence score of {state.probability_score * 100 if state.probability_score else 'N/A'}% based on comprehensive data analysis
- Prioritized {user_priorities} in the selection process
- Considered specific requirements for {state.office_size} office space

"""

    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    
    # Format final report with headers and sources
    state.running_summary = f"{report_header}\n## Detailed Analysis\n{state.running_summary}\n\n## Sources\n{all_sources}"
    
    return {"running_summary": state.running_summary}

# Helper function to generate a star rating for cities
def get_city_rating(rent, crime_rate, property_tax):
    """Generate a 1-5 star rating for a city based on key metrics."""
    # Lower values are better for these metrics
    rent_score = 5 - min(int(float(rent) / 20) if rent != 'N/A' else 2, 4)
    crime_score = 5 - min(int(float(crime_rate) / 2) if crime_rate != 'N/A' else 2, 4)
    tax_score = 5 - min(int(float(property_tax) / 1) if property_tax != 'N/A' else 2, 4)
    
    # Compute average and round to nearest integer
    avg_score = round((rent_score + crime_score + tax_score) / 3)
    return min(max(avg_score, 1), 5)  # Ensure score is between 1-5

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """LangGraph routing function that determines the next step in the research flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(config)
    
    # Debug logging
    print(f"RESEARCH LOOP STATUS: current={state.research_loop_count}, max={configurable.max_web_research_loops}")
    
    # Ensure at least one loop of web research always happens
    if state.research_loop_count == 0:
        print("FORCING first web research loop")
        return "web_research"
    
    # Continue research if we're under the max loops
    if state.research_loop_count < configurable.max_web_research_loops:
        print(f"CONTINUING research: loop {state.research_loop_count} of {configurable.max_web_research_loops}")
        return "web_research"
    else:
        print(f"FINALIZING summary after {state.research_loop_count} research loops")
        return "finalize_summary"

# Replace the SimpleProgressHandler class with this updated version
class SimpleProgressHandler(BaseCallbackHandler):
    """A minimal callback handler that only tracks basic progress information.
    
    This handler is designed to be maximally compatible with different 
    LangChain/LangGraph versions by implementing only the essential interfaces.
    """
    
    def __init__(self, update_func=None):
        """Initialize with optional update function."""
        super().__init__()
        self.update_func = update_func
        self._raise_error = False

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False
    
    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False
    
    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False
    
    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False
    
    @property
    def raise_error(self) -> bool:
        """Whether to raise errors."""
        return self._raise_error
    
    @raise_error.setter
    def raise_error(self, value: bool) -> None:
        """Setter for raise_error."""
        self._raise_error = value
    
    def on_llm_start(self, *args, **kwargs):
        """Handle LLM start events."""
        self._log_event("LLM processing", 30)
    
    def on_chain_start(self, *args, **kwargs):
        """Handle chain start events."""
        self._log_event("Chain processing", 50)
    
    def on_chain_end(self, *args, **kwargs):
        """Handle chain end events."""
        self._log_event("Chain completed", 70)
    
    def on_llm_end(self, *args, **kwargs):
        """Handle LLM end events."""
        self._log_event("LLM completed", 90)
    
    def _log_event(self, stage, percentage):
        """Log a progress event."""
        if self.update_func:
            try:
                self.update_func({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "readable_name": stage,
                    "percentage": percentage,
                    "message": f"{stage} ({percentage}%)"
                })
            except Exception as e:
                print(f"Error in simple progress handler: {e}")

# Also update the ProgressCallbackHandler class to use the same pattern
class ProgressCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking progress through LangGraph nodes.
    
    This class properly implements the LangChain callback protocol.
    """
    
    def __init__(self, update_func=None):
        """Initialize the callback handler.
        
        Args:
            update_func: Optional function to call with progress updates
        """
        super().__init__()
        self.update_func = update_func
        self._raise_error = False
        
        # Node mapping for human-readable names and percentages
        self.node_names = {
            "generate_query": "Generating search query",
            "web_research": "Researching market data",
            "summarize_sources": "Analyzing gathered information",
            "reflect_on_summary": "Evaluating analysis completeness",
            "finalize_summary": "Creating final report",
            "END": "Completed analysis"
        }
        
        self.progress_percentages = {
            "generate_query": 20,
            "web_research": 40,
            "summarize_sources": 60,
            "reflect_on_summary": 80,
            "finalize_summary": 95,
            "END": 100
        }
    
    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False
    
    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False
    
    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False
    
    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False
    
    @property
    def raise_error(self) -> bool:
        """Whether to raise errors."""
        return self._raise_error
    
    @raise_error.setter
    def raise_error(self, value: bool) -> None:
        """Setter for raise_error."""
        self._raise_error = value

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain starts running."""
        try:
            # Extract node name from run_id
            run_id = kwargs.get("run_id", "unknown")
            if isinstance(run_id, str) and "_" in run_id:
                node_name = run_id.split("_")[-1]
                self._log_progress(node_name)
        except Exception as e:
            print(f"Warning: Error in progress callback: {str(e)}")
            # Don't fail the whole process due to logging issues
    
    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain ends running."""
        pass
    
    def _log_progress(self, node_name):
        """Log progress for a specific node."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Get readable name and percentage
        readable_name = self.node_names.get(node_name, node_name)
        percentage = self.progress_percentages.get(node_name, 0)
        
        # Create the progress message
        progress_message = f"[{timestamp}] {readable_name} ({percentage}%)"

        # Print to terminal for debugging
        print(f"\n>>> PROGRESS UPDATE: {progress_message}\n")
        
        # If there's a specific output file defined, write to it
        progress_file = os.environ.get("PROGRESS_LOG_FILE")
        if progress_file:
            with open(progress_file, "a") as f:
                f.write(f"{progress_message}\n")
        
        # If an update function was provided, call it with the progress data
        if self.update_func:
            update_data = {
                "timestamp": timestamp,
                "node": node_name,
                "readable_name": readable_name,
                "percentage": percentage,
                "message": progress_message
            }
            self.update_func(update_data)

# Add a new minimal callback handler after the SimpleProgressHandler class
class MinimalCallback(BaseCallbackHandler):
    """Ultra-minimal callback handler with no property overrides.
    
    This class inherits from BaseCallbackHandler but doesn't override
    any properties, making it maximally compatible with different versions.
    """
    
    def __init__(self, update_func=None):
        super().__init__()
        self.update_func = update_func
    
    def on_llm_start(self, *args, **kwargs):
        self._log_event("Processing", 30)
    
    def on_chain_start(self, *args, **kwargs):
        self._log_event("Processing", 50)
    
    def on_chain_end(self, *args, **kwargs):
        self._log_event("Processing", 70)
    
    def on_llm_end(self, *args, **kwargs):
        self._log_event("Processing", 90)
    
    def _log_event(self, stage, percentage):
        """Log a progress event."""
        # Only use terminal logging to avoid UI issues
        print(f"\n>>> PROGRESS: {stage} ({percentage}%)\n")
        
        # Only call update_func if provided
        if self.update_func:
            try:
                self.update_func({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "readable_name": stage,
                    "percentage": percentage,
                    "message": f"{stage} ({percentage}%)"
                })
            except Exception as e:
                print(f"Error in minimal callback: {e}")

# Update the create_progress_callback function
def create_progress_callback(update_func=None, simplified=False, minimal=False):
    """Create a callback handler for tracking LangGraph progress.
    
    Args:
        update_func: Optional function to call with progress updates
        simplified: Use simplified handler with better compatibility
        minimal: Use minimal handler with no property overrides
        
    Returns:
        A callback handler that can be passed to the graph
    """
    if minimal:
        return MinimalCallback(update_func)
    if simplified:
        return SimpleProgressHandler(update_func)
    return ProgressCallbackHandler(update_func)

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

# Create the graph with progress tracking
graph = builder.compile()

# Update exports
__all__ = ["graph", "create_progress_callback", "ProgressCallbackHandler", "SimpleProgressHandler", "MinimalCallback"]