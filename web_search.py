from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(region="vn-en", max_results=3)

search = DuckDuckGoSearchRun(api_wrapper=wrapper)

def searching(query: str):
    result = search.run(query)
    return [{
        "text": result,
        "score": 1.0,
        "metadata": {},
        "rank": 1
    }]
