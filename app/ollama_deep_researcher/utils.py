import os
import httpx
import requests
from typing import Dict, Any, List, Union, Optional
from markdownify import markdownify
from langsmith import traceable
from tavily import TavilyClient


from langchain_community.utilities import SearxSearchWrapper

def get_config_value(value: Any) -> str:
    """
    Convert configuration values to string format, handling both string and enum types.
    
    Args:
        value (Any): The configuration value to process. Can be a string or an Enum.
    
    Returns:
        str: The string representation of the value.
        
    Examples:
        >>> get_config_value("tavily")
        'tavily'
        >>> get_config_value(SearchAPI.TAVILY)
        'tavily'
    """
    return value if isinstance(value, str) else value.value

def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.
    
    Iteratively removes all occurrences of content enclosed in thinking tokens.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text

def deduplicate_and_format_sources(
    search_response: Union[Dict[str, Any], List[Dict[str, Any]]], 
    max_tokens_per_source: int, 
    fetch_full_page: bool = False
) -> str:
    """
    Format and deduplicate search responses from various search APIs.
    
    Takes either a single search response or list of responses from search APIs,
    deduplicates them by URL, and formats them into a structured string.
    
    Args:
        search_response (Union[Dict[str, Any], List[Dict[str, Any]]]): Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
        max_tokens_per_source (int): Maximum number of tokens to include for each source's content
        fetch_full_page (bool, optional): Whether to include the full page content. Defaults to False.
            
    Returns:
        str: Formatted string with deduplicated sources
        
    Raises:
        ValueError: If input is neither a dict with 'results' key nor a list of search results
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.
    
    Creates a simple bulleted list of search results with title and URL for each source.
    
    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
        
    Returns:
        str: Formatted string with sources as bullet points in the format "* title : url"
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

def fetch_raw_content(url: str) -> Optional[str]:
    """
    Fetch HTML content from a URL and convert it to markdown format.
    
    Uses a 10-second timeout to avoid hanging on slow sites or large pages.
    
    Args:
        url (str): The URL to fetch content from
        
    Returns:
        Optional[str]: The fetched content converted to markdown if successful,
                      None if any error occurs during fetching or conversion
    """
    try:                
        # Create a client with reasonable timeout
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        return None

    
@traceable
def tavily_search(query: str, fetch_full_page: bool = True, max_results: int = 5, api_key: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search the web using the Tavily API and return formatted results with real estate focus.
    
    Uses the TavilyClient to perform searches, with a bias toward real estate market data
    and commercial property information.
    
    Args:
        query (str): The search query to execute
        fetch_full_page (bool, optional): Whether to include raw content from sources.
                                         Defaults to True.
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        api_key (Optional[str], optional): Tavily API key. If None, will try to get from environment.
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Search response containing real estate focused results
        
    Raises:
        Exception: If Tavily API key is not available or if the API request fails
    """
    # Get API key from parameter or environment
    tavily_api_key = api_key or os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        print("Warning: No Tavily API key found. Using fallback search.")
        # Return a mock response as fallback
        return {
            "results": [
                {
                    "title": "Search Failed: No Tavily API Key",
                    "url": "https://example.com",
                    "content": "Unable to perform search due to missing Tavily API key. Please check your configuration.",
                    "raw_content": "No API key provided"
                }
            ]
        }
    
    try:
        # Initialize Tavily client with explicit API key
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        print(f"Performing Tavily search with query: {query}")
        
        # Use search_context parameter to focus on real estate information
        response = tavily_client.search(
            query, 
            max_results=max_results, 
            include_raw_content=fetch_full_page,
            search_depth="advanced",  # More comprehensive search
            include_domains=[
                "loopnet.com", 
                "crexi.com", 
                "costar.com", 
                "jll.com", 
                "cbre.com", 
                "cushmanwakefield.com",
                "colliers.com",
                "bls.gov",
                "census.gov",
                "data.gov",
                "citydata.com"
            ]  # Include major CRE websites and government data sources
        )
        
        print(f"Tavily search completed successfully with {len(response.get('results', []))} results")
        return response
        
    except Exception as e:
        print(f"Error during Tavily search: {str(e)}")
        # Return a mock response with the error
        return {
            "results": [
                {
                    "title": "Search Error",
                    "url": "https://example.com/error",
                    "content": f"An error occurred during search: {str(e)}",
                    "raw_content": f"Search error: {str(e)}"
                }
            ]
        }

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int = 0) -> Dict[str, Any]:
    """
    Search the web using the Perplexity API with real estate focus.
    
    Uses the Perplexity API to perform searches with the 'sonar-pro' model,
    focusing on commercial real estate market trends and data.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int, optional): The loop step for perplexity search
                                                     (used for source labeling). Defaults to 0.
  
    Returns:
        Dict[str, Any]: Search response containing commercial real estate focused results
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    # Add real estate specific system instructions
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are a commercial real estate research assistant. Search the web and provide factual, data-driven information about commercial real estate markets, economic trends, and industry-specific factors. Focus on recent data (last 1-2 years) when available. Include relevant statistics, market trends, and business climate indicators."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()  # Raise exception for bad status codes
    
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])
    
    # Return first citation with full content, others just as references
    results = [{
        "title": f"Commercial Real Estate Analysis {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Commercial Real Estate Analysis {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}