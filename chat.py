from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from upload_data import cargar_documentos, crear_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Inicializamos el asistente al iniciar el backend
class Question(BaseModel):
    pregunta: str

ruta_archivo = "src/InformacionWebCoopebombas.pdf"

# Definimos modelos e instancias solo una vez
llm = Ollama(model="llama3.2")
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    embedding_function=embed_model,
    persist_directory="chroma_db_coopebombas",
    collection_name="coopebombasInfo1"
)

# Cargamos documentos si es necesario
if len(vectorstore.get()['ids']) == 0:
    docs = cargar_documentos(ruta_archivo)
    vectorstore = crear_vectorstore(docs)

# Definimos el recuperador y la plantilla de prompt
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

custom_prompt_template = """You are the virtual assistant of the company Coopebombas and your name is CoopebomasAI, 
    always be polite and respectful, answer and greet in a natural language. You will be provided with a context to answer the user's questions. 
    If you don't find a coherent answer, try to answer with your normal knowledge as an LLM or look it up on the web.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else but in spanish.
    Helpful answer:
"""
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Creamos la cadena de preguntas y respuestas
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

@app.post("/preguntar")
async def preguntar(data: Question):
    try:
        # Usamos la entrada de pregunta directamente
        respuesta = qa.invoke({"query": data.pregunta})
        
        # Procesamos metadata
        metadata = []
        for doc in respuesta['source_documents']:
            metadata.append(('page: ' + str(doc.metadata['page']), doc.metadata['file_path']))
        
        return {"respuesta": respuesta['result'], "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
