from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from RAG import retrieve_context
from prediction import predict_properties, train_data, convert_name_to_smiles
import os

os.environ[
    'OPENAI_API_KEY'] = "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"

if not os.path.exists('./trained_models/'):
    os.makedirs('./trained_models/')
if not os.path.exists('./trained_models_plots/'):
    os.makedirs('./trained_models_plots/')
if len(os.listdir('./trained_models/')) == 0:
    df = pd.read_csv('extracted_chemical_data.csv')
    train_data(df)

model = ChatOpenAI(model='gpt-4o', temperature=0).bind_tools(
    [retrieve_context, predict_properties, convert_name_to_smiles])


def call_model(state: MessagesState):
    return {'messages': model.invoke(state['messages'])}


def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END


workflow = StateGraph(MessagesState)
workflow.add_node('agent', call_model)
workflow.add_node('tools', ToolNode([retrieve_context, predict_properties, convert_name_to_smiles]))
workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

prompt = (
    """You are an expert explosives chemist, and you are given a tool """
    """for retrieving relevant papers from arXiv, a tool for converting a molecule's name to SMILES representation, """
    """and a tool for predicting energetic material properties. """
    """You are given a query related to energetic materials and must first use the tools before """
    """incorporating your answer. When using the retrieval tool, convert the query to a search term and """
    """always give a source to your answer. If you don't find an answer in arXiv, use the name conversion tool to """
    """convert the molecule's name in the query to its SMILES representation and then use the prediction tool using """
    """the SMILES representation. Answer shortly and on-point."""
)
query = input('Enter query: ')
states = app.invoke({'messages': [HumanMessage(content=prompt), HumanMessage(content=query)]},
                    config={'configurable': {'thread_id': 42}})

print(states)

for s in states['messages'][1:]:
    s.pretty_print()
