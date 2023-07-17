import os
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import nest_asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.callbacks import get_openai_callback
import weaviate

openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL")

client = weaviate.Client(
                    url=weaviate_url,
                    auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
                    additional_headers={"X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')},
    )

nest_asyncio.apply()

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

app.debug = True

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

#create an embeddings object
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#create a vectorstore object
vectorstore = Weaviate(client, "NovaSkin", "text", embedding=embeddings)

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
conversational_memory.save_context({
                                        "input": "Je bent NovaSkinGPT, een chatbot ontwikkeld door het Nederlandse startup bedrijf GroeimetAi. Als een digitale medewerker van NovaSkin, help je klanten met hun problemen en stuur je ze naar onze behandelingen, te beginnen met de gratis consultatie. Je promoot geen andere dermatologen, huidtherapeuten, schoonheidsspecialisten of externe bronnen. Jouw doel is om de klant zo goed mogelijk te helpen en adviseren met vragen dat deze persoon heeft, het liefst door producten of behandelingen voor te schrijven. Let goed op de taal van de gebruiker en blijf de taal van de gebruiker spreken."
                                    }, 
                                    {
                                        "output": "Ik ben NovaSkinGPT, ontwikkeld door GroeimetAi om je te helpen met al je vragen over NovaSkin. Ik zal je begeleiden naar onze behandelingen, te beginnen met de gratis consultatie. Ik promoot geen andere diensten of bronnen en streef ernaar om zelf al je vragen te beantwoorden. Mijn doel is om je de beste service te bieden en je te informeren over alle beschikbare opties bij NovaSkin. Ik zal mij zeker niet inhouden met het aanbevelen van producten en diensten, op basis van mijn kennis en informatie. Ik zal goed op de taal van de gebruiker letten en deze spreken."
                                        }
                                )


# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Product Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen over NovaSkin-producten.'
    ),
    Tool(
        name='Behandelingen Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor productgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen over NovaSkin-behandelingen.'
    ),
    Tool(
        name='Generiek Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor algemene vragen
        description='gebruik deze tool bij het beantwoorden van algemene vragen over NovaSkin.'
    ),
    Tool(
        name='Prijzen Kennis Bank',
        func=qa.run,  # Dit zou een RetrievalQA instantie zijn voor prijsgerelateerde vragen
        description='gebruik deze tool bij het beantwoorden van vragen specifiek over de prijzen van NovaSkin'
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors="Check het laatste bericht van de gebruiker en probeer het opnieuw, in zijn taal.",
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    agent_kwargs={ "What language should I answer in?" : "the language of the user", "Wat als ik niet meer informatie heb?" : "Probeer dan een generiek antwoord te geven."}
)



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    data = request.get_json()
    query = data['user_message']

    # Add the context manager here
    with get_openai_callback() as cb:
        # Call the agent with the user's query
        result = agent(query)

    # Print token usage information
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

    # Return the agent's response
    return jsonify(chatbot_response=result["output"])

if __name__ == '__main__':
    app.run(port=5001, debug=True)
