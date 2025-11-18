import os

from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langgraph.types import Command
from dotenv import load_dotenv

# Global
load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v3.2",
    base_url=os.getenv("LLM_API_URL") or "",
    api_key=os.getenv("LLM_API_KEY") or "",
    temperature=0,
)


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_result: list[str] | None
    customer_history: dict | None

    # Generated content
    draft_reponse: str | None
    message: list[str] | None


# Read and classify nodes
def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    # In production, this would connect to your email service;
    return {
        "messages": [
            HumanMessage(content=f"Processing email: {state['email_content']}")
        ]
    }


def classify_intent(
    state: EmailAgentState,
) -> Command[
    Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]
]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state["email_content"]}
    From: {state["sender_email"]}

    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response directly as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification
    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        goto = "human_review"
    elif classification["intent"] in ["question", "feature"]:
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification}, goto=goto)
