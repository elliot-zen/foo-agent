import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=os.getenv("LLM_API_URL") or "",
    api_key=os.getenv("LLM_API_KEY") or "",
    temperature=0,
)


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("钙CT评分与高胆固醇有何关系？")

print(output)
