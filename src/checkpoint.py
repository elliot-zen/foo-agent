import operator
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], operator.add]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


if __name__ == "__main__":
    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": 1}}
    graph.invoke({"foo": ""}, config)

    for cp in list(graph.get_state_history(config)):
        print(cp)
