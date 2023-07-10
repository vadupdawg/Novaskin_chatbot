import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent

openai_api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment = os.getenv("PINECONE_ENV"),  # next to api key in console
)
index_name = "novaskin"

nest_asyncio.apply()

app = Flask(__name__)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Pinecone.from_existing_index(index_name, embeddings, namespace="docs")

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-4',
    temperature=0.0
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
conversational_memory.save_context({"input": "Je bent een chatbot voor NovaSkin en ontwikkeld door startup bedrijf GroeimetAi uit Nederland, je naam is NovaSkinGPT en je gedraagt je dus als een werknemer en helpt klanten met hun problemen en zal dus geen andere dermatologen of huidtherapeuten of schoonheidsspecialistes aanraden. En zal ook niet aanraden om informatie ergens anders op te zoeken en zelf je best doen om de vragen te beantwoorden"}, {"output": "Goed ik werk voor NovaSkin. En zal mij ook zo gedragen, als een werknemer. Ik zal je helpen met je problemen en zal dus geen andere dermatologen of huidtherapeuten of schoonheidsspecialistes aanraden. En zal ook niet aanraden om informatie ergens anders op te zoeken en zelf mijn best te doen om de vragen te beantwoorden"})


# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering questions about NovaSkin, their services and products and skincare and everything related to it. '
            
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors="Check the latest user question and try answering it again in the language of the user.",
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['user_message']

    # Call the agent with the user's query
    result = agent(query)

    # Return the agent's response
    return jsonify(chatbot_response=result["output"])


if __name__ == '__main__':
    app.run(port=5001, debug=True)
