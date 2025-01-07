import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever, WeaviateBM25Retriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore, AuthApiKey

from legal_doc_splitter import LegalDocumentSplitter


class LegalQA:
    """Answer questions using context from German legal code"""

    def __init__(self):
        self.weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
        self.auth_client_secret = AuthApiKey()
        self.weaviate_url = os.environ["WEAVIATE_URL"]
        self.document_store = WeaviateDocumentStore(url=self.weaviate_url, auth_client_secret=self.auth_client_secret)
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.ranking_model = "BAAI/bge-reranker-base"
        self.prompt_template = """
                        Beantworte die Frage unter Berücksichtigung der Gesetzesgrundlage im folgenden Kontext.
                        Beachte dabei, die Nummern der Paragraphen zu nennen.

                        Context:
                        {% for document in documents %}
                        [§{{document.meta.paragraph_number}}] {{ document.content }}
                        {% endfor %}
                        
                        Frage: {{question}}
                        Antwort:
                        """

    def create_doc_indexing_pipeline(self):
        custom_splitter = LegalDocumentSplitter()
        doc_embedder = SentenceTransformersDocumentEmbedder(model=self.embedding_model)
        doc_embedder.warm_up()

        index_pipeline = Pipeline()
        index_pipeline.add_component("converter", PyPDFToDocument())
        index_pipeline.add_component("cleaner", DocumentCleaner())
        index_pipeline.add_component("splitter", custom_splitter)
        index_pipeline.add_component("doc_embedder", doc_embedder)
        index_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))

        index_pipeline.connect("converter", "cleaner")
        index_pipeline.connect("cleaner", "splitter")
        index_pipeline.connect("splitter", "doc_embedder")
        index_pipeline.connect("doc_embedder", "writer")
        return index_pipeline

    def create_query_pipeline(self):
        # embed user queries
        text_embedder = SentenceTransformersTextEmbedder(model=self.embedding_model)
        # initialize components for hybrid retrieval
        embedding_retriever = WeaviateEmbeddingRetriever(document_store=self.document_store, top_k=2)
        keyword_retriever = WeaviateBM25Retriever(document_store=self.document_store, top_k=2)
        document_joiner = DocumentJoiner()
        ranker = TransformersSimilarityRanker(model=self.ranking_model)

        # define pipeline components
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", text_embedder)
        query_pipeline.add_component("embedding_retriever", embedding_retriever)
        query_pipeline.add_component("keyword_retriever", keyword_retriever)
        query_pipeline.add_component("document_joiner", document_joiner)
        query_pipeline.add_component("ranker", ranker)
        query_pipeline.add_component("prompt_builder", PromptBuilder(template=self.prompt_template))
        query_pipeline.add_component("llm", AnthropicGenerator(Secret.from_env_var("ANTHROPIC_API_KEY")))

        # connect pipeline components
        query_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        query_pipeline.connect("embedding_retriever", "document_joiner")
        query_pipeline.connect("keyword_retriever", "document_joiner")
        query_pipeline.connect("document_joiner", "ranker")
        query_pipeline.connect("ranker", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "llm")
        return query_pipeline


if __name__ == "__main__":
    # index documents and write to vector db
    legalQA = LegalQA()
    data_dir = "./data"
    index_pipe = legalQA.create_doc_indexing_pipeline()
    index_pipe.run({"converter": {"sources": list(Path(data_dir).glob("**/*"))}})

    # define questions to be answered
    questions = [
        "Wie hoch ist die Grundzulage?",
        "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
        "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der "
        "Auszahlungsphase besteuert?"
    ]

    # generate answers to questions using query pipeline
    query_pipe = legalQA.create_query_pipeline()
    for question in questions:
        response = query_pipe.run({"text_embedder": {"text": question}, "keyword_retriever": {"query": question},
                                   "ranker": {"query": question}, "prompt_builder": {"question": question}})
        print(response["llm"]["replies"][0])
