from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    query: str
    query_type: Optional[str]
    retrieved_docs: Optional[List[str]]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    is_valid: Optional[bool]
    confidence_score: Optional[float]
    retry_count: Optional[int]