# creating an agentic ai Infowhiz by langchain
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGCHAIN_API_KEY "]= os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")   
os.environ["SERP_API_KEY"]= os.getenv("SERP_API_KEY")

#1st retriever tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv)
#2nd retriever tool
from langchain_community.utilities import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun

scholar = GoogleScholarAPIWrapper()
scholar_tool = GoogleScholarQueryRun(api_wrapper=scholar)
#3rd retriever tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
tools = [arxiv_tool, scholar_tool, wiki_tool]
#for context memory


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState,StateGraph
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated,TypedDict
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder


from langgraph.prebuilt import create_react_agent



from langchain_core.messages import SystemMessage,trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import HumanMessage,AIMessage

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(
    title="Nami API",
    description="Nami is an AI agent that can help you with a variety of tasks, including research, coding, and general questions.",
    version="0.1",
)

class Data(BaseModel):
    user:str
    thread_id:str
    query: str


@app.post("/nami")
async def model(data: Data):
    response = nami_agent(thread_id=data.thread_id, query=data.query)
    think = list(response.split("</think>"))[0].replace("<think>","")
    answer = list(response.split("</think>"))[1]
    return {"think":think, "answer":answer}




class AgentConfig:
    default_language: str = "English"
    max_tokens_ctx: int = 1500

class WrappedAgent:
    def __init__(self, model, tools,prompt, cfg: AgentConfig = AgentConfig()):
        self.cfg = cfg
        self.memory = MemorySaver()

        def pre_model_hook(state):
            trimmed = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=self.cfg.max_tokens_ctx,
                start_on="human",
                include_system=True,
                allow_partial=True,
                end_on=("human","tool"),
            )
            return {"messages": trimmed}

        self.agent = create_react_agent(
            model, tools,
            checkpointer=self.memory,
            prompt=prompt,                 # or messages/state modifier per version
            pre_model_hook=pre_model_hook  # trims per call
        )

    def __call__(self, user_query: str,extra_state=None, language="English", thread_id=""  ) -> dict[str, any]:
        config = {"configurable": {"thread_id": thread_id or "thread-"}}  # inject a real id generator in prod
        if language:
            config["configurable"]["language"] = language
        input_state = {"messages": [("user", user_query)]}
        if extra_state:
            input_state.update(extra_state)
        result = self.agent.invoke(input_state, config)
        final_msg = result["messages"][-1]
        answer ={
            "answer": getattr(final_msg, "content", final_msg),
            "thread_id": config["configurable"]["thread_id"]
        } 
        return answer 

# Expose as a tool callable by another agent
def infowhiz_tool(user_query, language="English", thread_id=" ") -> str:
    prompt = ChatPromptTemplate.from_messages(
                    [
                    (   "system", 
                """You are a helpful research assistant that helps users find information about a given topic in a specified language(if mentioned in the last message query otherwise use english).
                     You have access to the following tools: Arxiv, Google Scholar, Wikipedia.Use one of it which is most suitable for the query if You can't find anything useful use another tool.answer for general questions without using tools
                                        """
         ),
         MessagesPlaceholder(variable_name="messages"),


        ]
        )
    infowhiz = WrappedAgent(
        model = init_chat_model("llama-3.1-8b-instant", model_provider="groq"),
        tools = [arxiv_tool, scholar_tool, wiki_tool],
        prompt = prompt,
    )
    answer = infowhiz(user_query, language=language, thread_id=thread_id)["answer"]
    return answer

#thert is no streaming in here, in future make sure it has one

def chat_buddy_tool(user_query, language="English", thread_id=" ") -> str:
    prompt = ChatPromptTemplate.from_messages(
                    [
                    (   "system", 
                """You are a friendly chat buddy that helps users in a friendly manner in a specified language(if mentioned in the last message query otherwise use english).
                     You have access to the following tools: Arxiv, Google Scholar, Wikipedia.Use one of it which is most suitable for the query if You can't find anything useful use another tool.answer for general questions without using tools
                                        """
         ),
         MessagesPlaceholder(variable_name="messages"),


        ]
        )
    chat_buddy = WrappedAgent(
        model = init_chat_model("qwen/qwen3-32b", model_provider="groq"),
        tools = [],
        prompt = prompt,
    )
    return chat_buddy(user_query, language=language, thread_id=thread_id)["answer"]


def Task_master_tool(user_query, language="English", thread_id=" ") -> str:
    prompt = ChatPromptTemplate.from_messages(
                    [
                    (   "system", 
                """You are a task master that helps users to break down their tasks into smaller tasks in a specified language(if mentioned in the last message query otherwise use english).
                     You have access to the following tools: Arxiv, Google Scholar, Wikipedia.Use one of it which is most suitable for the query if You can't find anything useful use another tool.answer for general questions without using tools
                                        """
         ),
         MessagesPlaceholder(variable_name="messages"),


        ]
        )
    task_master = WrappedAgent(
        model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq"),
        tools = [],
        prompt = prompt,
    )
    return task_master(user_query, language=language, thread_id=thread_id)["answer"]


def coder_tool(user_query, language="English", thread_id=" ") -> str:
    prompt = ChatPromptTemplate.from_messages(
                    [
                    (   "system", 
                """You are a coding assistant that helps users to write code in a specified language(if mentioned in the last message query otherwise use english).
                     You have access to the following tools: Arxiv, Google Scholar, Wikipedia.Use one of it which is most suitable for the query if You can't find anything useful use another tool.answer for general questions without using tools
                                        """
         ),
         MessagesPlaceholder(variable_name="messages"),


        ]
        )
    coder = WrappedAgent(
        model = init_chat_model("gemma2-9b-it", model_provider="groq"),
        tools = [],
        prompt = prompt,
    )
    return coder(user_query, language=language, thread_id=thread_id)["answer"]

def nami_agent(thread_id, query):
    model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    # 1) Build the chat prompt
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a classifier that selects the most suitable tool for the given query and its retrieved context.\n"
     "You have access to the following tools and their toolcodes(number 0-3):\n"
     "- chat_buddy_tool: casual chat, emotional support, small talk. - 0\n"
     "- infowhiz_tool: research, academic questions, finding information. -1\n"
     "- Task_master_tool: task breakdown, planning, scheduling, execution. -2\n"
     "- coder_tool: writing or explaining code (Python/C++ etc.). 3\n\n"
     "Use the context to refine routing if it changes intent or adds constraints.\n\n"
     "Return a Python-like string with two elements:\n"
     '<tool_code>,"<possibly refined query>"\n'
     "Only use one of: chat_buddy_tool-0, infowhiz_tool-1, Task_master_tool-2, coder_tool-3."
    ),
    ("human", "Query: {query}")
    ])

    # 2) Format to messages
    prompt_value = prompt.invoke({"query": query})
    messages = prompt_value.to_messages()

    # 3) Call the model directly
    response = model.invoke(messages)
      # in prod, use a real id generator
    if response.content[0]=="1":
        print("infowhiz")
        answer = infowhiz_tool(user_query=response.content, thread_id=thread_id)
    elif response.content[0]=="0":
        print("chat buddy")
        answer = chat_buddy_tool(user_query=response.content, thread_id=thread_id)
    elif response.content[0]=="2":
        answer = Task_master_tool(user_query=response.content, thread_id=thread_id)
    elif response.content[0]=="3":
        answer = coder_tool(user_query=response.content, thread_id=thread_id)
    return answer
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    

