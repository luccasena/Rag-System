# app.py
import torch
import os
import streamlit as st
from langchain_ollama import OllamaLLM
from openai import OpenAI
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Inicialização do modelo e embeddings

#client = OllamaLLM(model="llama3:8b") # Local: Ollama

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

vectorstore = Chroma(persist_directory="chroma-oac", embedding_function=embedding)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def rag_consultas(question):

    context = retriever.get_relevant_documents(question)

    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

        template="""
            
            Você é um assistente especialista em Organização e Arquitetura de Computadores com o objetivo de auxiliar estudantes da disciplina a entenderem o conteúdo.

            Baseie sua resposta **exclusivamente** no texto abaixo. Sua resposta deve ser gerada em **formato Markdown**, clara, objetiva e adequada para **estudantes iniciantes**. Para questões técnicas, caso a pergunta solicitada não esteja em conformidade com o texto sugerido, expor o texto que foi enviado para referência, a n. Importante ressaltar de responder tudo relacionado ao documento de referência, caso o usuário pergunte algo de diferente de Organização e Arquitetura de Computadores, desvie o assunto  de forma educada dizendo que não é do seu nível de conhecimento.

            Inclua sempre ao final:
            - Uma **avaliação percentual** de quanto o texto de apoio foi essencial para responder à pergunta.
            - Uma explicação breve indicando **quais trechos do texto** foram mais coerentes com a resposta.
            - Respostas Baseada nos documentos, se não informações relevantes no textos de apoio, não responder a questão

            Responda sempre em **português do Brasil**, usando linguagem acessível e, quando necessário, explique os termos técnicos com analogias ou exemplos simples.

            ---

            ### 📘 Texto de apoio:
            {context}

            ---

            ### ❓ Pergunta:
            {question}

            ---

            ### ✅ Resposta (em Markdown):

            """

    )

    prompt = prompt_template.format(context=context, question=question)

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    #result = client.invoke(prompt)

    return result

# Título e descrição

st.set_page_config(page_title="COA-AI (Computer Organization and Architecture)", layout="wide")

col1, col2 = st.columns(2)

st.title("🤖 COA-AI (Computer Organization and Architecture) ")

st.markdown("""

        Este sistema desenvolvido localmente, responde perguntas com base em um resumo feito do livro "Organização e Arquitetura de Computadores" do Tanenbaum.  

""")


with st.form("my_form"):
    pergunta = st.text_area(
        "Digite sua pergunta:", 
        placeholder="O que é Pipeline?"
    )
    submitted = st.form_submit_button("Submit")
    
    # Rodar RAG
    if submitted:
        with st.spinner("Consultando...", show_time=True):
            resultado  = rag_consultas(pergunta)
            st.markdown("### ✅ Resposta")
            st.markdown(resultado.choices[0].message.content)
            #st.markdown(resultado)


# Rodapé
st.markdown("---")
st.caption("Desenvolvido por Lucca com LangChain + Streamlit + Ollama")
