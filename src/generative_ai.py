from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv 
import json
from prompt import  PROMPT, KEY_OPENAI

import os



class Askcsv():
    def __init__(self, filename: str, temperature=0) -> None:
        self.filename = filename
        self.temperature = temperature
        self.df = pd.read_csv(self.filename)
        
    def csv_tool(self):

        return create_pandas_dataframe_agent(OpenAI(temperature=self.temperature, openai_api_key=KEY_OPENAI), self.df, verbose=True)


    def ask_agent(self, agent, query):
        """
        Query an agent and return the response as a string.

        Args:
            agent: The agent to query.
            query: The query to ask the agent.

        Returns:
            The response from the agent as a string.
        """
        # Prepare the prompt with query guidelines and formatting

        prompt = PROMPT + query

        response = agent.run(prompt)
 
        return self.encode_response(response)

    def encode_response(self, response):

        # response = response.replace('\'', '"')
 
        try:
            response_json = json.loads(response)
            return response_json

        except Exception as error:
   
            return  {"error": error, "response": response}
        
    def execute(self, query: str):
        agent = self.csv_tool()
        response = self.ask_agent(agent=agent, query=query)

        return response



# class_ask = Askcsv("data/Salary_Data.csv")

# # response = class_ask.execute("What is the best job title by salary")
# response = class_ask.execute("give me the 5 job title higher salary")


# print(response)

