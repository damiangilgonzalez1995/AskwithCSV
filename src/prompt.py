import os
PROMPT = """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
        {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
        {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        4. If a histogram chart is more appropriate, your reply should look like this:
        {"hist": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn't need a chart or table, your response should be:
        {"answer": "Your answer goes here"}
        Note that, in this case, the response must not contain double quotes
        For example:
        {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
        {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 

        """
        
   



# KEY_OPENAI = "yourkey"
KEY_OPENAI = os.environ['OPENAI_API_KEY']

# or you can use API_KEY = os.environ['OPENAI_API_KEY']