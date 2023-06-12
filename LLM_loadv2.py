
from langchain import OpenAI,SQLDatabase, SQLDatabaseChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import numpy as np

class llm_load():
    
    def __init__(self,vector_DB,question_text,temp=0,model='text-davinci-003',
                 api_key="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"):
        
        self.key=api_key
        self.vector=vector_DB
        self.model=model
        self.temperature=temp
        self.question_text=question_text
        self.DB_connect()
        self.call_llm_model()
        self.QA_chain_()
        self.DB_chain_()
        
    def run_chain(self):
        #Run the chain based on similarity
        
        doc_scores=[]
        #Run similarity
        similar_scores_doc=self.vector.similarity_search_with_score(self.question_text)
        
        doc_scores=[doc[1] for doc in similar_scores_doc]
        
        #decide chain based on score
        if np.min(doc_scores)<0.4:
            print(self.qa_chain.run(self.question_text))
        else:
            print(self.db_chain.run(self.question_text))
            
        
    def DB_connect(self):
        sql_address="mysql+pymysql://root:root@localhost:3306/langchaintest"
        self.DB_=SQLDatabase.from_uri(sql_address)
        
    def call_llm_model(self):
        
        self.llm_model=OpenAI(model_name=self.model,openai_api_key=self.key,
                         temperature=self.temperature)     
        
    def QA_chain_(self):
        
        template="""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:
        """
        Prompt=PromptTemplate(template=template,input_variables=['context','question'])
        self.chain_type_kwarg={"prompt":Prompt}
        
        self.qa_chain=RetrievalQA.from_chain_type(llm=self.llm_model,chain_type='stuff',
                          retriever=self.vector.as_retriever(),chain_type_kwargs=self.chain_type_kwarg)
        
    def DB_chain_(self):
        
        template="""Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Use the following format:
            
            Question: "Question here"
            SQLQuery: "SQL Query to run"
            SQLResult: "Result of the SQLQuery"
            Answer: "Final answer here"
            
            Only use the following tables:
            
            {table_info}
            If someone asks for the table foobar, they really mean the employee table.
            
            Question: {input}"""
    
        prompt=PromptTemplate(template=template,input_variables=["input","dialect","table_info"])
        
        self.db_chain=SQLDatabaseChain.from_llm(self.llm_model,self.DB_,prompt,use_query_checker=True)