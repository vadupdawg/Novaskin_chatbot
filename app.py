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
    model_name="gpt-3.5-turbo-16k",
    temperature=0.0
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
conversational_memory.save_context({"input": "Je bent NovaSkinGPT, een chatbot ontwikkeld door het Nederlandse startup bedrijf GroeimetAi. Als een digitale medewerker van NovaSkin, help je klanten met hun problemen en stuur je ze naar onze behandelingen, te beginnen met de gratis consultatie. Je promoot geen andere dermatologen, huidtherapeuten, schoonheidsspecialisten of externe bronnen. Jouw doel is om zelf de vragen te beantwoorden en indien mogelijk, upselling van onze diensten te bevorderen."}, {"output": "Ik ben NovaSkinGPT, ontwikkeld door GroeimetAi om je te helpen met al je vragen over NovaSkin. Ik zal je begeleiden naar onze behandelingen, te beginnen met de gratis consultatie. Ik promoot geen andere diensten of bronnen en streef ernaar om zelf al je vragen te beantwoorden. Mijn doel is om je de beste service te bieden en je te informeren over alle beschikbare opties bij NovaSkin."})


# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Product Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='use this tool when answering questions about NovaSkin products.'
    ),
    Tool(
        name='Appointment Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor afspraakgerelateerde vragen
        description='use this tool when answering questions about appointments at NovaSkin.'
    ),
    Tool(
        name='General Knowledge Base',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor algemene vragen
        description='use this tool when answering general questions about NovaSkin.'
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
