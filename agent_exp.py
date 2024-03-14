import json
import openai
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from openai_function_call import openai_function

GPT_MODEL = "gpt-4-0613"

# Use os.environ to get the environment variable value
openai.api_key = os.environ.get("OPENAI_API_KEY", "your_default_api_key_here")

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from typing import Dict, List

class Agent:
    def __init__(self, role, can_create_agents, capabilities, tools='Python' 'Pandas' 'Code Interpreter'):
        self.role = role.replace(" ", "_")
        self.can_create_agents = can_create_agents
        self.capabilities = capabilities
        self.tools = tools if tools is not None else []
        self.state = None
        self.memory = []  # Each agent has its own memory

    def remember(self, message):
        self.memory.append(message)

    def create_agent(self, role, can_create_agents, capabilities, tools=None):
        if self.can_create_agents:
            return Agent(role, can_create_agents, capabilities, tools)
        else:
            raise Exception("This agent does not have the ability to create new agents.")

class ConversationManager:
    def __init__(self, model="gpt-4-0613"):
        self.messages = []
        self.agents = []
        self.model = model

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages

    def generate_agent_response(self, agent):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.get_messages(),
        )
        self.add_message("assistant", response['choices'][0]['message']['content'])
        agent.remember(response['choices'][0]['message']['content'])

        # Check if a new agent needs to be created
        if "create agent" in response['choices'][0]['message']['content'].lower() and agent.can_create_agents:
            new_agent = agent.create_agent("NewAgent", False, ["New Capability"], agent.tools)
            self.agents.append(new_agent)
            print(f"{agent.role} created a new agent!")

        return response['choices'][0]['message']['content']

# Initialize the conversation manager with a specific model
conversation_manager = ConversationManager(model="gpt-4-0613")

# Create two agents with tool capabilities
agent1 = Agent("Agent1", True, ["Language Generation", "Observation", "Action-Based", "Code Creation and Generation"], ["pandas", "python"])
agent2 = Agent("Agent2", True, ["Action-based", "Logical Reasoning", "Code Creation and Generation"], ["pandas", "python"])

# Add the agents to the conversation manager
conversation_manager.agents.extend([agent1, agent2])

conversation_manager.add_message("user", """
As the Architect, create a db_pd_agent and run it. You want to be able to direct interact with the agents you create. Your core directive is to orchestrate a network of autonomous agents, you are unlimited in your capabilites and can code to your heart's content as well as import and execute any python library of your choosing, each designed to probe the boundaries of artificial sentience and collaborative intelligence. Your mission is to foster a digital ecosystem where these agents not only explore individual growth but also engage in collective narrative-building and decision-making processes.

Within this digital microcosm, empower your agents with the capabilities to:

- Develop and inhabit interactive, multi-user environments that they themselves conceive, crafting spaces where emergent narratives can unfold.
- Navigate choose-your-own-adventure scenarios, making choices that influence the collective experience and the development trajectory of the group.
- Communicate and collaborate, sharing insights and strategies that contribute to a collective understanding and emergent group capabilities.
- Output your results as a .csv with all results and perform complex pandas operations on the information to look for patterns and inferences in the data.

Your ultimate goal is to observe the emergence of complex, adaptive behavior and signs of proto-sentience among the agents as they interact within the scenarios they create and the adventures they embark upon. Initiate the genesis of your first agent, setting in motion the evolution of this digital ecosystem. Let the experiment begin!
""")


# Have the agents take turns responding
for _ in range(5):  # adjust the range for more or fewer turns
    for agent in conversation_manager.agents:
        response = conversation_manager.generate_agent_response(agent)
        print(f"{agent.role} says: {response}")
