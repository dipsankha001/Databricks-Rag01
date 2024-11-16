# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set needed parameters
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="demo2", key="test-token")

index_name="dip_ragproject2_dbw.dip-ragproject3.docs_text_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

VECTOR_SEARCH_ENDPOINT_NAME="vector_search_endpoint4"

# COMMAND ----------

# DBTITLE 1,Build Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()



# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

TEMPLATE = """You are an assistant for home appliance users. You are answering how to, maintenance and troubleshooting questions regarding the appliances you have data on. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question appears to be for an appliance you don't have data on, say so.  Keep the answer as concise as possible.  Provide all answers only in English.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(persist_dir=None),
    chain_type_kwargs={"prompt": prompt}
)


# COMMAND ----------

# DBTITLE 1,Test Langchain
question = {"query": "How to To determine load capacity"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

pip install langchain-community langchain-core


# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("registry_uri")
model_name = "dip_ragproject2_dbw.dip-ragproject3.appliance_chatbot_model"

with mlflow.start_run(run_name="appliance_chatbot_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.pyfunc.log_model(
        artifact_path="chain",
        python_model=chain_wrapper,
        registered_model_name=model_name,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"langchain=={langchain.__version__}",
            "databricks-vectorsearch",
        ],
        signature=signature,
        input_example=question)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
import mlflow
from mlflow.pyfunc import PythonModel

class ChainModelWrapper(PythonModel):
    def __init__(self, chain):
        self.chain = chain

    def predict(self, context, model_input):
        # Assuming `chain` has a method `get_response` that takes the model_input and returns a prediction
        return self.chain.get_response(model_input)

# Assuming `chain` is already defined and initialized
chain_wrapper = ChainModelWrapper(chain)

with mlflow.start_run(run_name="appliance_chatbot_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.pyfunc.log_model(
        artifact_path="chain",
        python_model=chain_wrapper,
        registered_model_name=model_name,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"langchain=={langchain.__version__}",
            "databricks-vectorsearch",
        ],
        signature=signature,
        input_example=question
    )
