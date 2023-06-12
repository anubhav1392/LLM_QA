#Only contains one llm and handle one chain 


from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class llm_load:
    
    def __init__(self,vector_DB,question_text,temp=0,model='text-davinci-003',
                 api_key="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"):
        
        self.key=api_key
        self.vector=vector_DB
        self.model=model
        self.temperature=temp
        self.question_text=question_text
        self.prompt_temp()
        
    def call_llm_model(self):
        
        llm_model=OpenAI(model_name=self.model,openai_api_key=self.key,
                         temperature=self.temperature)     
        
        chain=RetrievalQA.from_chain_type(llm=llm_model,chain_type='stuff',
                          retriever=self.vector.as_retriever(),chain_type_kwargs=self.chain_type_kwarg)
    
        print(chain.run(self.question_text))
        
    def prompt_temp(self):
        
        template="""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:
        """
        Prompt=PromptTemplate(template=template,input_variables=['context','question'])
        self.chain_type_kwarg={"prompt":Prompt}
        
    