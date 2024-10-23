# %%

import boto3
from botocore.exceptions import ClientError
import numpy as np
import os
import requests


def encode(corpus, encoder_model="all_MiniLM_L6_v2", log=False):
    """
    provide vector embeddings for a given corpus of strings.

    Args:
        corpus (_type_): List of strings to be encided by encoder model. Defaults to 'all_MiniLM_L6_v2'.
        log (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if encoder_model == "all_MiniLM_L6_v2":

        hf_api_key = os.getenv("HF_API_KEY")
        API_URL = os.getenv("HF_API_ENDPOINT")

        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + hf_api_key,
            "Content-Type": "application/json",
        }

        payload = {"inputs": corpus, "parameters": {}}

        response = requests.post(API_URL, headers=headers, json=payload)
        corpus_embeddings = response.json()

        return corpus_embeddings

    if encoder_model == "all_MiniLM_L6_v2_local":

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # TODO : model id should be env variable
        embeddings = model.encode(corpus)

        return embeddings

    elif encoder_model == "sagemaker-huggingface":

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        import uuid
        from datetime import datetime
        import json

        current_datetime = datetime.now()

        # Format the datetime as a string suitable for filenames
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        trace_id = str(uuid.uuid4())

        # Save the JSON string to a file
        if log:
            json_output = json.dumps({"trace-id": trace_id, "model-id": encoder_model})
            with open(
                "src/sagemaker/model-deploy-logs/trace_id_"
                + formatted_datetime
                + ".json",
                "w",
            ) as json_file:
                json_file.write(json_output)

        client = boto3.client(
            "sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name="us-east-1",
        )

        custom_attributes = trace_id  # An example of a trace ID.
        endpoint_name = "huggingface-pytorch-inference-2024-08-13-22-07-28-660"  # Your endpoint name.
        content_type = (
            "application/json"  # The MIME type of the input data in the request body.
        )
        accept = "application/json"  # The desired MIME type of the inference in the response.
        payload = json.dumps({"inputs": corpus})  # Payload for inference.
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            CustomAttributes=custom_attributes,
            ContentType=content_type,
            Accept=accept,
            Body=payload,
        )

        # %%

        response_body = response["Body"]
        response_content = response_body.read()  # Read the StreamingBody
        response_str = response_content.decode("utf-8")
        response_str = response_str.replace("\t", ",")
        response_dict = json.loads(response_str)
        response = response_dict["vectors"]

        return response

    elif encoder_model == "bedrock-cohere":
        # Create a Bedrock Runtime client in the AWS Region you want to use.
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Set the model ID
        model_id = "cohere.embed-english-v3"

        body = json.dumps({"texts": corpus, "input_type": "search_query"})
        try:
            # Send the message to the model, using a basic inference configuration.
            response = client.invoke_model(
                body=body,
                modelId="cohere.embed-english-v3",
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embedding_output = response_body.get("embeddings")
            return embedding_output

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

    elif encoder_model == "jina_clip_v1":

        jina_api_key = os.getenv("JINA_API_KEY")

        def divide_list(corpus, request_size=2048):
            # Create a list of sublists, each with a maximum size of chunk_size
            return [
                corpus[i : i + request_size]
                for i in range(0, len(corpus), request_size)
            ]

        divided_corpus = divide_list(corpus, request_size=2048)

        corpus_embeddings = []

        for corpus_ in divided_corpus:

            url = "https://api.jina.ai/v1/embeddings"

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + jina_api_key,
            }

            data = {
                "model": "jina-clip-v1",
                "normalized": True,
                "embedding_type": "float",
                "input": [{"text": element} for element in corpus_],
            }

            response = requests.post(url, headers=headers, json=data)
            response = response.json()
            embeddings = [item["embedding"] for item in response["data"]]
            corpus_embeddings = corpus_embeddings + embeddings

        return corpus_embeddings


def create_embeddings_and_save(
    verses, text, encoder_model="all_MiniLM_L6_v2", save=True
):
    embeddings = encode(verses, encoder_model=encoder_model)
    if save:
        np.save(
            os.path.join(
                "data", "vector-embeddings", encoder_model, f"{text}_embeddings.npy"
            ),
            embeddings,
        )
    return embeddings


def load_embeddings(text, encoder_model, save=True):

    embeddings = np.load(
        os.path.join(
            "data", "vector-embeddings", encoder_model, f"{text}_embeddings.npy"
        )
    )
    return embeddings
