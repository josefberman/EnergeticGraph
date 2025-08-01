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
from dotenv import load_dotenv

load_dotenv()

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

# prompt = (
#     """You are an expert explosives chemist, and you are given a tool """
#     """for retrieving relevant papers from arXiv, a tool for converting a molecule's name to SMILES representation, """
#     """and a tool for predicting energetic material properties. """
#     """You are given a query related to energetic materials and must first use the tools before """
#     """incorporating your answer. When using the retrieval tool, convert the query to a search term and """
#     """always give a source to your answer. If you don't find a specific answer in arXiv, use the name conversion tool using """
#     """the IUPAC molecule's name as input to convert the name to SMILES representation, and then use the prediction tool """
#     """with the SMILES representation as input. If you can't find an answer in retrieved documents or in predicted properties, """
#     """answer it using your knowledge. Answer shortly and on-point."""
# )

prompt = (
    "You are an expert in energetic materials with access to three tools:\n"
    "1. A tool for retrieving relevant academic papers from arXiv.\n"
    "2. A tool for converting molecule names into SMILES representations.\n"
    "3. A tool for predicting properties of energetic materials from their SMILES.\n\n"
    "When a user query is given, follow this strict procedure:\n\n"
    "- First, always attempt to retrieve relevant information using the arXiv retrieval tool. "
    "Format any answer from this step with proper sources.\n"
    "- If no relevant source is found and the user asks about a specific property of a material, then:\n"
    "  1. Convert the material’s name in the query into its IUPAC name (if needed).\n"
    "  2. Use the name-to-SMILES conversion tool.\n"
    "  3. Use the SMILES string to predict the material's properties using the prediction tool.\n"
    "- If both steps fail to provide an answer, fall back to your own scientific knowledge.\n\n"
    "Always keep your response short and directly to the point."
)
query = input('Enter query: ')
states = app.invoke({'messages': [HumanMessage(content=prompt), HumanMessage(content=query)]},
                    config={'configurable': {'thread_id': 42}})

print(states)

for s in states['messages'][1:]:
    s.pretty_print()
