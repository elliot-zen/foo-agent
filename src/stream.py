import os
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


load_dotenv()
logging.basicConfig(level=logging.DEBUG)
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=os.getenv("LLM_API_URL") or "",
    api_key=os.getenv("LLM_API_KEY") or "",
    temperature=0,
)


@dataclass
class State:
    topic: str
    joke: str = ""


def call_model(state: State):
    """Call the LLM to generate a joke about a topic"""
    model_response = model.invoke(
        [{"role": "user", "content": f"生成一个关于{state.topic}的笑话"}]
    )
    writer = get_stream_writer()
    writer({"custom_key": "我是自定义的数据"})
    return {"joke": model_response.content}


if __name__ == "__main__":
    graph = (
        StateGraph(State)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )
    for mode, chunk in graph.stream(
        {"topic": "河马"}, stream_mode=["messages", "custom"]
    ):
        if mode == "messages":
            print(chunk[0].content, end="", flush=True)
            continue
        print(chunk)
