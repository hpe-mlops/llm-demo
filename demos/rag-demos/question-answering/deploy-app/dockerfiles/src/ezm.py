import os
import glob
import base64
import requests
import subprocess
import mlflow
import time

from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

domain = None
username = None
password = None
predictor_image = None


# llm
llm_inference_service_name = None
llm_predictor_name = None
llm_predictor_image = None
llm_predictor_cpu_request = None
llm_predictor_cpu_limit = None
llm_predictor_memory_request = None
llm_predictor_memory_limit = None
llm_predictor_gpu_request = None
llm_predictor_gpu_limit = None
llm_transformer_name = None
llm_transformer_image = None
llm_transformer_cpu_request = None
llm_transformer_cpu_limit = None
llm_transformer_memory_request = None
llm_transformer_memory_limit = None
llm_transformer_gpu_request = None
llm_transformer_gpu_limit = None

def config_environ():
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "10.96.0.0/12, .local"

    os.environ['MLFLOW_TRACKING_TOKEN'] = get_token()
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ['MLFLOW_TRACKING_TOKEN']
    os.environ["AWS_SECRET_ACCESS_KEY"] = "s3"
    os.environ["AWS_ENDPOINT_URL"] = 'http://local-s3-service.ezdata-system.svc.cluster.local:30000'
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ["AWS_ENDPOINT_URL"]
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.mlflow.svc.cluster.local:5000"

def get_token():

    global domain
    token_url = f"https://keycloak.{domain}/realms/UA/protocol/openid-connect/token"
    print(domain)
    data = {
        "username" : username,
        "password" : password,
        "grant_type" : "password",
        "client_id" : "ua-grant",
    }   
    token_responce = requests.post(token_url, data=data, allow_redirects=True, verify=False)
    token = token_responce.json()["access_token"]
    print("TOKEN: {token}")
    return token

def check_object_ready(obj, name):
    result = subprocess.run(["kubectl", "get", obj+"/"+name, "-n", username],  capture_output=True)
    output = result.stdout.decode("utf-8")

    
    #print(result)
    #print(output)

    if not name in output:
        return False
    if "True" in output.split("\n")[1]:
        return True
        
def wait_for_ready(obj, name): # 5 * 12 * 10 = 10M
    complete = False
    for i in range(12 * 10):
        complete = check_object_ready(obj, name)
        if complete:
            break
        time.sleep(5)
        
    if(complete):
        print("K8S Object Ready: {name}")
    else:
        print("K8S Object Timeout: {name}")

def delete_object(obj, name):
    subprocess.run(["kubectl", "delete", obj+"/"+name, "-n", username])


## 01.create-vectorstore ##
def load_doc(fn):
    loader = TextLoader(fn)
    doc = loader.load()
    return doc
	
def load_docs(source_dir: str) -> list:
    """Load all documents in a the given directory."""
    fns = glob.glob(os.path.join(source_dir, "*.txt"))
    docs = []
    for i, fn in enumerate(tqdm(fns, desc="Loading documents...")):
        docs.extend(load_doc(fn))

    return docs
	
def process_docs(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """Load the documents and split them into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts
	
def hugging_face_embeddings():
    embeddings_model = "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=embeddings_model)
	
def storing_chroma(texts, embeddings):
    db = Chroma.from_documents(texts, embeddings, persist_directory=f"{os.getcwd()}/db")
    db.persist()
    return db
	
def test_query(db):
    query = "How can I create a cgroup?"
    matches = db.similarity_search(query); matches
    print(matches)

## 02.serve-vectorstore ##
def encode_base64(message: str):
    encoded_bytes = base64.b64encode(message.encode('ASCII'))
    return encoded_bytes.decode('ASCII')

def get_or_create_experiment(exp_name):
    """Register an experiment in MLFlow.
    
    args:
      exp_name (str): The name of the experiment.
    """
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        raise RuntimeError(f"Failed to set the experiment: {e}")

def logging_vector_store_artifact():
    # Create a new MLFlow experiment or re-use an existing one
    get_or_create_experiment('question-answering')
    # Log the Chroma DB files as an artifact of the experiment
    print(f"{os.getcwd()}/db")
    mlflow.log_artifact(f"{os.getcwd()}/db")
    # Retrieve the URI of the artifact
    uri = mlflow.get_artifact_uri("db")
    return uri

def apply_vectorstore(uri):
    isvc = """
apiVersion: v1
kind: Secret
metadata:
  name: minio-secret
type: Opaque
data:
  MINIO_ACCESS_KEY: {0}
  MINIO_SECRET_KEY: {1}

---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vectorstore
spec:
  predictor:
    containers:
    - name: kserve-container
      image: {2}
      imagePullPolicy: Always
      resources:
        requests:
          memory: "2Gi"
          cpu: "500m"
        limits:
          memory: "2Gi"
          cpu: "500m"
      args:
      - --persist-uri
      - {3}
      env:
      # If you are running behind a proxy, uncomment the following lines and replace the values with your proxy URLs.
      - name: HTTP_PROXY
        value: {4}
      - name: HTTPS_PROXY
        value: {5}
      - name: NO_PROXY
        value: .local
      - name: MLFLOW_S3_ENDPOINT_URL
        value: {6}
      - name: TRANSFORMERS_CACHE
        value: /src
      - name: SENTENCE_TRANSFORMERS_HOME
        value: /src
      - name: MINIO_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            key: MINIO_ACCESS_KEY
            name: minio-secret
      - name: MINIO_SECRET_KEY
        valueFrom:
          secretKeyRef:
            key: MINIO_SECRET_KEY
            name: minio-secret
""".format(encode_base64(os.environ["AWS_ACCESS_KEY_ID"]),
           encode_base64(os.environ["AWS_SECRET_ACCESS_KEY"]),
           predictor_image,
           uri,
           os.environ["HTTP_PROXY"],
           os.environ["HTTPS_PROXY"],
           os.environ["MLFLOW_S3_ENDPOINT_URL"])
    with open("vectorstore-isvc.yaml", "w") as f:
        f.write(isvc)
    
    subprocess.run(["kubectl", "apply", "-f", "vectorstore-isvc.yaml"])

    wait_for_ready("inferenceservice","vectorstore")


## 04.serve-llm ##
def apply_llm():
    isvc = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {0}
spec:
  predictor:
    timeout: 600
    containers:
      - name: {1}
        image: {2}
        imagePullPolicy: Always
        # If you are running behind a proxy, uncomment the following lines and replace the values with your proxy URLs.
        env:
        - name: HTTP_PROXY
          value: {3}
        - name: HTTPS_PROXY
          value: {4}
        - name: NO_PROXY
          value: .local
        resources:
          requests:
            memory: "{7}"
            cpu: "{5}"
          limits:
            memory: "{8}"
            cpu: "{6}"
  transformer:
    timeout: 600
    containers:
      - image: {10}
        imagePullPolicy: Always
        resources:
          requests:
            memory: "{13}"
            cpu: "{11}"
          limits:
            memory: "{14}"
            cpu: "{12}"
        name: {9}
        args: ["--use_ssl"]
""".format(llm_inference_service_name,
           llm_predictor_name,
           llm_predictor_image,
           os.environ["HTTP_PROXY"],
           os.environ["HTTPS_PROXY"],
           llm_predictor_cpu_request,
           llm_predictor_cpu_limit,
           llm_predictor_memory_request,
           llm_predictor_memory_limit,
           llm_transformer_name,
           llm_transformer_image,
           llm_transformer_cpu_request,
           llm_transformer_cpu_limit,
           llm_transformer_memory_request,
           llm_transformer_memory_limit)
    with open("llm-isvc.yaml", "w") as f:
        f.write(isvc)
    subprocess.run(["kubectl", "apply", "-f", "llm-isvc.yaml"])
    
    wait_for_ready("inferenceservice", "llm")