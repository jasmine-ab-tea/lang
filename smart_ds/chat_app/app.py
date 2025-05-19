import streamlit as st
from chat import agent, HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolMessage

st.set_page_config(
    page_title="Experiment Analysis Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Experiment Analysis Chatbot")
st.markdown("""
This chatbot helps you analyze experiment data. You can ask questions about:
- Treatment effects across different countries
- Statistical significance of experiments
- Data patterns and trends
- And more!
""")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the experiment data"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the agent
            messages = [HumanMessage(content=prompt)]
            response = agent.invoke({"messages": messages}, {"configurable": {"thread_id": "1"}})
            
            # Display each message from the agent
            # Only display messages since the last user question
            last_user_index = next(
                (i for i, msg in reversed(list(enumerate(response["messages"]))) if isinstance(msg, HumanMessage)), 
                -1
            ) + 1
            for message in response["messages"][last_user_index:]:
                if message.content != "":
                    st.markdown(f"🤖🤖 {message.content}")
                elif hasattr(message, 'tool_calls'):
                    for tool_call in message.tool_calls:
                        st.markdown(f"🔧 🔧 Using tool: {tool_call['name']}")

    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response["messages"][-1].content)}) 