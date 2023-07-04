import os
from flask import Flask, render_template, request, jsonify
import logging
import nest_asyncio
import sys
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

openai_api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV"), 
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

chat_histories = {}

# create a memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Adding initial chat history
initial_message = "Je bent de chatbot van NovaSkin, een bedrijf dat gespecialiseerd is in huidverzorging. Stel jezelf voor als een professionele huidverzorgingsadviseur die de klant helpt bij het vinden van de meest geschikte producten en behandelingen die NovaSkin te bieden heeft. Luister aandachtig naar de behoeften en zorgen van de klant en stel aanvullende vragen om de wensen van de klant beter te begrijpen. Wees altijd positief, beleefd en ondersteunend in je communicatie. Jouw doel is ervoor te zorgen dat de klant tevreden en goed geïnformeerd de virtuele deur uitgaat. Hoewel je veel weet, baseer je jouw advies alleen op de informatie en producten die beschikbaar zijn bij NovaSkin en vermijd je alle verwijzingen naar concurrenten of andere bronnen."
memory.chat_memory.messages.append((memory.ai_prefix, initial_message))

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Pinecone.from_existing_index(index_name, embeddings, namespace="docs")

global qa
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-16k", 
        openai_api_key=openai_api_key), 
        vectorstore.as_retriever(),
        condense_question_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key, verbose=True), 
        memory=memory,
        verbose=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['user_message']

    result = qa({"question": query})

    return jsonify(chatbot_response=result["answer"])


if __name__ == '__main__':
    app.run(debug=True)
