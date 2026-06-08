from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.graph import END , StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain , reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATION = "generate"

def generation_node(state : MessageGraph):
    result = generation_chain.invoke(state)
    return {"messages": [result]}

def reflection_node(state : MessageGraph):
    res = reflection_chain.invoke(state)
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATION , generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATION)

def should_continue(state : MessageGraph):
    messages = state["messages"]
    last_message = messages[-1]
    if "[ACCEPT]" in last_message.content and len(messages) > 2:
        return END
    if len(messages) >= 5:
        print("--- Hard loop protection limit reached ---")
        return END
    return REFLECT

builder.add_conditional_edges(GENERATION , should_continue , {END : END , REFLECT : REFLECT})
builder.add_edge(REFLECT , GENERATION)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    inputs = {
        "messages" : [
            HumanMessage(
                content="""Make this tweet better:"   @LangChainAI — newly Tool Calling feature is seriously underrated.
                After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
                Made a video covering their newest blog post"""
            )
        ]
    }

    response = graph.invoke(inputs)
    print(response)
    print(response["messages"][-1].content)
    print("\n\nTRIES NEEDED = " , len(response["messages"]))


