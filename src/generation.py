# %%

import boto3
from botocore.exceptions import ClientError
import os


def get_oracle_response(user_query, agent="oracle", local="localdocker"):
    # Create a Bedrock Runtime client in the AWS Region you want to use.

    if local == "locahost":

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    elif local == "localdocker":
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    else:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    # Set the model ID
    model_id = "meta.llama3-70b-instruct-v1:0"

    if agent == "oracle":
        # Start a conversation with the user message.
        instruction = """[INST]You are an oracle who will be fed a user query and several 
                        verses which are semantically connected to that query.
                        Make sure your response explores the theme shared by user with wisdom
                        making reference to the passages that were given together with the user query
                        make sure you interweave the passages into a coherent message.   
                        Also make sure you conclude your message before reaching 400 words.[/INST]"""

    if agent == "scientist":
        # Start a conversation with the user message.
        instruction = """[INST]You are a scientist who will be fed a user query and several 
                        verses which are semantically connected to that query.
                        Make sure your response explores the themes presented by the local highest centrality subgraph in an analytical manner
                        making reference to the passages that were brought together by embedding model and a semantic similarity index
                        make sure you try to find out what is the shared meaning accross the different verses, 
                        tell me what is the main message that is shared by the verses   
                        Also make sure you conclude your message before reaching 400 words.[/INST]"""

    if agent == "exegete":
        # Start a conversation with the user message.
        instruction = """[INST]take the four verses which are the most central verses in this text
                        extract what you think is the main message the those verses carry. 
                        if you receive a verse 2, 3 or 4 times, it means that verse is specially central 
                        the task here is to try to understando the core meaning of what is being in said, 
                        in terms of what concepts give the text its tone and general message.
                        Make sure you are concise and clear in your response, and respectful in regards to the source text
                        Also make sure you conclude your message before reaching 400 words.[/INST]"""

    user_message = instruction + user_query

    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
            additionalModelRequestFields={},
        )
        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
