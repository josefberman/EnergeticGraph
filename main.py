from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from RAG import retrieve_context
from prediction import predict_properties, train_all_models
import os

os.environ['OPENAI_API_KEY'] = "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"

if not os.path.exists('./trained_models/'):
    os.makedirs('./trained_models/')
if not os.path.exists('./trained_models_plots/'):
    os.makedirs('./trained_models_plots/')
if not os.listdir('./trained_models/'):
    train_all_models()


model = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools([retrieve_context, predict_properties])


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
    """for retrieving relevant papers from arXiv, and a tool for predicting energetic material properties. """
    """You are given a query related to energetic materials and must first use the tools before """
    """incorporating your answer. When using the retrieval tool, convert the query to a search term and """
    """always give a source to your answer. If you don't find an answer in arXiv, use the prediction tool by """
    """converting the material's name in the query to a SMILES representation. Answer shortly and on-point."""
)
query = input('Enter query: ')
states = app.invoke({'messages': [HumanMessage(content=prompt), HumanMessage(content=query)]},
                    config={'configurable': {'thread_id': 42}})

print(states)

for s in states['messages'][1:]:
    s.pretty_print()
