from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import os

os.environ['NVIDIA_API_KEY'] = "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"

