import getpass
import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph.graph import MessagesState
from typing_extensions import Literal
import statsmodels.api as sm
from statsmodels.formula.api import ols

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )

# Create a SQLAlchemy engine
engine = create_engine('sqlite:///experiment_data.db')

# Load the experiment_data.csv into the database before initializing SQLDatabase
df = pd.read_csv('/Users/yixin/Projects/lang/smart_ds/experiment_data.csv')
df.to_sql('experiment_data', con=engine, if_exists='replace', index=False)

# Initialize the SQLDatabase with the engine
db = SQLDatabase(engine)

# Initialize Language model
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Define tools
@tool
def compute_treatment_effect(x: str):
    """
    Given a string representing a dataframe with two columns (first element of each tuple is index) such as 
    '[(1, 0.2054792820232115, 0), (2, 0.2723859937913791, 0), (3, 0.2624978915182918, 0)]', 
    compute the treatment effect of the data. The treatment effect is defined as the difference-in-means
    of the metric value (second element of each tuple) with respect to the treatment assignment 
    (third element of each tuple). 

    Returns:
        dict: A dictionary containing both the treatment effect and p-value.
    """
    # Convert the string x into a list of tuples
    data = eval(x)
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=['index', 'metric', 'experiment_group'])
    
    # Calculate p-value using the formula from the non-tool version
    formula = "metric ~ experiment_group"  # Using 'metric' as the dependent variable and 'experiment_group' as the independent variable
    model = ols(formula, data=df).fit()
    p_value = model.pvalues['experiment_group']
    
    return {
        "treatment_effect": float(model.params['experiment_group']),
        "p_value": float(p_value)
    }

@tool
def generate_and_execute_sql(question):
    """
    Given an input human question in English, create a syntactically correct sqlite query to
    run to help find the answer
    """

    # Annotations must have the type and can optionally include a default value and description (in that order).
    question: Annotated[str, ..., "Question"]

    system_message = """
    Given an input question, create a syntactically correct {dialect} query to
    run to help find the answer. Unless the user specifies in his question a
    specific number of examples they wish to obtain, always limit your query to
    at most {top_k} results. You can order the results by a relevant column to
    return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a the
    few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema
    description. Be careful to not query for columns that do not exist. Also,
    pay attention to which column is in which table.

    Only use the following tables:
    {table_info}
    """

    user_prompt = "Question: {input}"

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )

    class QueryOutput(TypedDict):
        """Generated SQL query."""
        query: Annotated[str, ..., "Syntactically valid SQL query."]

    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 100,
            "table_info": db.get_table_info(),
            "input": question,
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    query = structured_llm.invoke(prompt)
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return execute_query_tool.invoke(query)


# Augment the LLM with tools
tools = [generate_and_execute_sql, compute_treatment_effect]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Define Node
class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    result: str
    answer: str

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    system_messgage_conent = """
    You are a helpful data scientist tasked with analyzing experiment data
    Tool Usage Instructions:
    - Use the 'generate_and_execute_sql' tool when you need to query the database and get some data for analysis
    - Use the 'compute_treatment_effect' tool when you need to compute the treatment effect of data, this tool is usually called
    after obtaining some data from calling 'generate_and_execute_sql' tool
    """
    return {
        "messages": 
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=system_messgage_conent
                    )
                ]
                + state["messages"]
            )        
    }


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(State)
# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_edge(START, "llm_call")

agent_builder.add_node("environment", tool_node)
agent_builder.add_edge("environment", "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)

agent = agent_builder.compile()

def stream_graph_updates(user_input: str):
    for event in agent.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value)



user_message = "what is the treatment effect of experiment"
# stream_graph_updates(user_message)

# Invoke
messages = [HumanMessage(content=user_message)]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()