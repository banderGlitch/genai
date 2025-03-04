from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi
import re
from IPython.display import Image, display
import requests
import json
import os
from dotenv import load_dotenv
from urllib.parse import urlpars

openai_groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",  # or "gpt-3.5-turbo" for a less expensive option
    temperature=0,
    api_key=openai_groq_key
)
