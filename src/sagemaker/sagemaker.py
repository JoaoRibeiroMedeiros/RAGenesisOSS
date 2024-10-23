# %% SETUP


import sagemaker as sagemaker
import boto3

sess = sagemaker.Session()

# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()
 
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')


role = iam.get_role(RoleName='SagemakerExecutionRole')['Role']['Arn']
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
 
print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")
print(f"sagemaker session bucket: {sess.default_bucket()}")


repository = "sentence-transformers/all-MiniLM-L6-v2"
model_id=repository.split("/")[-1]
s3_location=f"s3://{sess.default_bucket()}/custom_inference/{model_id}/model.tar.gz"


# %% UPLOAD MODEL TO S3

# def upload_model_to_s3():

!git lfs install
!git clone https://huggingface.co/$repository
!cd $model_id
!git lfs pull
!mkdir -p $model_id/code/
!cp -r src/sagemaker/inference.py $model_id/code/
%cd $model_id
!tar zcvf model.tar.gz *
!aws s3 cp model.tar.gz $s3_location
%cd ..

# %%  DEPLOY MODEL TO SAGEMAKER


from sagemaker.huggingface.model import HuggingFaceModel
 
huggingface_model = HuggingFaceModel(
   model_data=s3_location,       # path to your model and script
   role=role,                    # iam role with permissions to create an Endpoint
   transformers_version="4.12",  # transformers version used
   pytorch_version="1.9",        # pytorch version used
   py_version='py38',            # python version used
)
 
# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
    )

# %%

# %%  MAKE INFERENCE

import uuid
from datetime import datetime
import json

current_datetime = datetime.now()

# Format the datetime as a string suitable for filenames
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
trace_id = str(uuid.uuid4())
json_output = json.dumps({"trace-id": trace_id, "model-id": model_id})

# Save the JSON string to a file
with open("src/sagemaker/model-deploy-logs/trace_id_"+formatted_datetime+".json", "w") as json_file:
    json_file.write(json_output)

client = boto3.client('sagemaker-runtime')

custom_attributes = trace_id  # An example of a trace ID.
endpoint_name = "huggingface-pytorch-inference-2024-08-13-22-07-28-660"                                       # Your endpoint name.
content_type = "application/json"                                        # The MIME type of the input data in the request body.
accept = "application/json"                                              # The desired MIME type of the inference in the response.
payload = json.dumps({"inputs": "Your input text here"})                                             # Payload for inference.
response = client.invoke_endpoint(
    EndpointName=endpoint_name, 
    CustomAttributes=custom_attributes, 
    ContentType=content_type,
    Accept=accept,
    Body=payload
    )

# %%   

response_body = response['Body']     
response_content = response_body.read()  # Read the StreamingBody
response_str = response_content.decode('utf-8')
response_str = response_str.replace('\t', ',')
response_dict = json.loads(response_str)
response = response_dict["vectors"]

print(response)

# %%                    # If model receives and updates the custom_attributes header 
                                             

data = {
  "inputs": "the mesmerizing performances of the leads keep the film grounded and keep the audience riveted .",
}
#%%

res = predictor.predict(data=data)
print(res)

#%%

predictor.delete_model()
predictor.delete_endpoint()
# %%
