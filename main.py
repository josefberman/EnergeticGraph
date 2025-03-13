from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from RAG import retrieve_context
from prediction import predict_properties
import os

os.environ['NVIDIA_API_KEY'] = "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"

model = ChatNVIDIA(model_name='meta/llama-3.3-70b-instruct', temperature=0).bind_tools(
    [retrieve_context, predict_properties],
    tool_choice='required')


def call_model(state: MessagesState):
    return {'messages': model.invoke(state['messages'])}


def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END


workflow = StateGraph(MessagesState)
workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode([retrieve_context, predict_properties]))
workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

prompt = (
    """You are an expert explosives chemist, and you are given a tool """
    """for retrieving relevant papers from arXive, and a tool for predicting energetic material properties. """
    """You are given a query related to energetic materials and must first use one of the tools before """
    """incorporating your answer. When using the retrieval tool, always give a source to your answer."""
    """Answer shortly and on-point."""
)
query = input('Enter query: ')
states = app.invoke({'messages': [HumanMessage(content=prompt), HumanMessage(content=query)]},
                    config={'configurable': {'thread_id': 42}})

print(states)

for s in states['messages'][1:]:
    s.pretty_print()
