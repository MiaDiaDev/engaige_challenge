import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore, AuthApiKey

weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
auth_client_secret = AuthApiKey()
weaviate_url = os.environ["WEAVIATE_URL"]

document_store = WeaviateDocumentStore(url=weaviate_url, auth_client_secret=auth_client_secret)
# load embedding model
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# TODO best cleaning/splitting strategies? bigger chunks? change embedding model?
pipeline = Pipeline()
pipeline.add_component("converter", PyPDFToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
pipeline.add_component("doc_embedder", doc_embedder)
pipeline.add_component("writer", DocumentWriter(document_store=document_store))

pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "doc_embedder")
pipeline.connect("doc_embedder", "writer")

data_dir = "./data"
pipeline.run({"converter": {"sources": list(Path(data_dir).glob("**/*"))}})

# embed user queries
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
# initialize retriever
# TODO optimize parameters?
retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=5)

# set prompt template
# TODO optimize?
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

# define RAG pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
query_pipeline.add_component("llm", AnthropicGenerator(Secret.from_env_var("ANTHROPIC_API_KEY")))

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "llm")

# ask questions
questions = [
    "Wie hoch ist die Grundzulage?",
    "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
    "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der "
    "Auszahlungsphase besteuert?"
]

for question in questions:
    response = query_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
    print(response["llm"]["replies"][0])

# TODO add citation of docs or more finegrained?

# TODO close connection with client.close()?
