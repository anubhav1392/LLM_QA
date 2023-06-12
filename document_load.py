from langchain.document_loaders import UnstructuredFileLoader,PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os

file_path=r'C:\Users\AnubhavSingh\Langchain Test Folder\document'


class doc_load:
    def __init__(self, file_name,api_key):
        
        self.fname=file_name
        self.api_key=api_key


    #Upload File
    def upload_file(self):
        
        if self.fname[-3:]=='txt':
            loader=UnstructuredFileLoader(os.path.join(file_path,self.fname))
            self.doc=loader.load()
            
            self.chunk_creation()
            embeddings=self.create_embeddings()
            return embeddings
            
        elif self.fname[-3:]=='pdf':
            loader=PyPDFLoader(os.path.join(file_path,self.fname))
            self.doc=loader.load_and_split()
            
            embeddings=self.create_embeddings()
            return embeddings
            
        else:
            print('This format is not supported currently')
            
        
           
    #Split file into chunks
    def chunk_creation(self):
        #Create Chunks       
        splitter=CharacterTextSplitter(chunk_overlap=0,chunk_size=1000)
        self.doc=splitter.split_documents(self.doc)

        
    #Create Embeddings
    def create_embeddings(self):
        #Create embeddings
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        embeddings=OpenAIEmbeddings()
        doc_embedding=Chroma.from_documents(self.doc, embeddings)
        
        return doc_embedding