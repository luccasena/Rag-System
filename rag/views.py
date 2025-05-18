import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

client = Ollama(model="llama3:8b")

embedding = OllamaEmbeddings(model="bge-m3")

persist_directory = "chroma"


def index(request):
    return render(request, 'index.html')

def enviar_pergunta(pergunta, contexto):
    try:
        prompt = f"""
        Pergunta: {pergunta}

        Utilize o seguinte contexto para responder a pergunta:
        {contexto}

        Sobre o agente: Você é OAC, um assistente de inteligência artificial especializado em Organização e Arquitetura de Computadores. Sua função é ajudar os usuários a encontrar informações relevantes e responder perguntas sobre o contexto. Seja sempre claro, conciso e preciso. Se o contexto não contiver informações suficientes para uma resposta confiável, informe que não sabe. Se alguem perguntar sobre os desenvolvedores do agente, fale que eles são constituídos por Lucca de Sena Barbosa, estudante de Ciência de Dados que cursa o 3º período de Ciência da Computação, e Leonardo Lucas de Brito Silva, estudante de Desenvolvimento Back-End que cursa o 3º período de Ciência da Computação.

        """
        resposta = client.invoke(prompt)
        return resposta

    except Exception as e:
        return f"Ocorreu um erro: {e}"


def rag_consultas(request):

    data = json.loads(request.body)
    pergunta = data.get("pergunta")

    vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    docs = vectorstore.similarity_search_with_score(pergunta, k=4)

    resposta = enviar_pergunta(pergunta, str(docs))

    return JsonResponse({
                            "resultado": resposta
                        })