
import os
import getpass
from galileo.openai import openai # The Galileo OpenAI client wrapper is all you need!
from galileo import galileo_context # Only for notebooks and persistent processes

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        prompt="Enter your OpenAI API key (required if using OpenAI): "
    )


client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = "Explain the following topic succinctly: Newton's First Law"
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": prompt}],
)
galileo_context.flush()
print(response.choices[0].message.content.strip())
print("View log stream in Galileo at https://app.galileo.ai/get-started/first-logstream/")
