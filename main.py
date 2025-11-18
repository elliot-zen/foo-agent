from langgraph.graph import END, START, MessagesState, StateGraph


def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}


def main():
    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()
    graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})


if __name__ == "__main__":
    main()
