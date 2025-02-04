from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model = 'mistral')
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100)
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],  # Explicitly define input variables
            template="""<s> [INST] You are an assistant for question-answering tasks.
                                    Use the following pieces of retrieved context to answer 
                                    the question. If you don't know the answer, say that you 
                                    don't know. Use three sentences maximum and keep the 
                                    answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_path: str):
        docs =  PyPDFLoader(file_path = pdf_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = FAISS.from_documents(documents = chunks, embedding = HuggingFaceEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type = 'similarity_score_threshold',
            search_kwargs = {
                "k":3,
                "score_threshold": 0.5,
            }
        )

        self.chain = ({"context" : self.retriever, "question": RunnablePassthrough()}
                        | self.prompt
                        | self.model
                        | StrOutputParser())
        
    def ask(self,query:str):
        if not self.chain:
            return "Pls add a PDF document first."
        
        return self.chain.invoke(query)
    
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None