from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from RAG import retrieve_context
import os

os.environ['NVIDIA_API_KEY'] = "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"

model = ChatNVIDIA(model_name='meta/llama-3.3-70b-instruct', temperature=0).bind_tools([retrieve_context])


def call_model(state: MessagesState):
    return {'messages': model.invoke(state['messages'])}


def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END


workflow = StateGraph(MessagesState)
workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode([retrieve_context]))
workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

states = app.invoke({'messages': [HumanMessage(content=input('Enter query: '))]},
                    config={'configurable': {'thread_id': 42}})

for s in states['messages']:
    s.pretty_print()
