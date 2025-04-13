from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class SummaryStateInput:
    """Input data structure for the commercial real estate recommendation system."""
    research_topic: str
    recommended_city: str
    recommended_region: str
    industry_type: str
    probability_score: float
    crime_rate: float
    property_tax: float
    rent_class: str
    office_size: str
    rent_value: float = 0
    user_priorities: str = ""
    all_city_probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class SummaryState:
    """State for the commercial real estate recommendation system."""
    research_topic: str
    recommended_city: str
    recommended_region: str
    industry_type: str
    probability_score: float
    crime_rate: float
    property_tax: float
    rent_class: str
    office_size: str
    questions: List[str] = field(default_factory=list)
    query: str = ""
    current_query: str = ""
    summary: str = ""
    rent_value: float = 0
    user_priorities: str = ""
    all_city_probabilities: Dict[str, float] = field(default_factory=dict)
    web_research_results: List[str] = field(default_factory=list)
    sources_gathered: List[str] = field(default_factory=list)
    research_loop_count: int = 0
    running_summary: str = ""
    search_query: str = ""


@dataclass
class SummaryStateOutput:
    """Output data structure for the commercial real estate recommendation system."""
    running_summary: str
    search_query: str = ""
    recommended_city: str = ""
    industry_type: str = ""
    rent_value: float = 0
    crime_rate: float = 0
    property_tax: float = 0 