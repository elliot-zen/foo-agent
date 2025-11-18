import operator
import os

from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing import Literal
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="deepseek-v3.2",
    base_url=os.getenv("LLM_API_URL") or "",
    api_key=os.getenv("LLM_API_KEY") or "",
    temperature=0,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`

    Args:
        a: First init
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`

    Args:
        a: First init
        b: Second int
    """
    return a + b


@tool
def devide(a: int, b: int) -> float:
    """Devide `a` and `b`

    Args:
        a: First init
        b: Second int
    """
    return a / b


tools = [add, multiply, devide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return END


def main():
    agent_builder = StateGraph(MessageState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

    agent_builder.add_edge("tool_node", "llm_call")

    agent = agent_builder.compile()

    # Show the agent
    # bytes = agent.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(bytes)

    # Invoke
    from langchain.messages import HumanMessage

    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
