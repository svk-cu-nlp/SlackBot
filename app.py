import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
import json

#For RAG

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
import textwrap

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

#Initializing Embedding Model
openai_embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

openai_llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-3.5-turbo-0125")
persist_directory = 'db_book'

# Useful it first to retrieve the bot user id. After single run store the id in the credentials file
# def get_bot_user_id():
#     """
#     Get the bot user ID using the Slack API.
#     Returns:
#         str: The bot user ID.
#     """
#     try:
#         # Initialize the Slack client with your bot token
#         slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
#         response = slack_client.auth_test()
#         return response["user_id"]
#     except SlackApiError as e:
#         print(f"Error: {e}")


# RAG pipeline 
def parse_pdf():
    book_texts = None
    if not os.path.exists(persist_directory):

        loader = PyPDFLoader("./handbook.pdf")
        book_pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        book_texts = text_splitter.split_documents(book_pages)
    
    return book_texts

    



def creating_vectorDB(splitted_text):
    book_vectordb = None
    embedding = openai_embeddings_model
    if not os.path.exists(persist_directory):
        book_vectordb = Chroma.from_documents(documents=splitted_text,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    else:
        # load from disk
        book_vectordb = Chroma(persist_directory="./db_book", embedding_function=embedding)
    
    return book_vectordb


def create_retriever(vector_db, query):
    if vector_db is None:
        return
    
    query_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "Sorry! Can't Answer the Question" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:
                  """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(llm=openai_llm,
                                  chain_type="stuff",
                                  retriever=query_retriever,
                                       chain_type_kwargs={
                                            "verbose": False,
                                            "prompt": prompt,

                                        },
                                  return_source_documents=False)
    llm_response = qa_chain(query)
    response = process_llm_response(llm_response)
    return response
    
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    wrapped_text_string = str(wrapped_text)

    return wrapped_text_string

def process_llm_response(llm_response):
    response = wrap_text_preserve_newlines(llm_response['result'])
    return response

def create_pair_json(query, response):
    pair = {
        "query": query,
        "response": response
    }
    return json.dumps(pair)

def my_function(query):
    book_text = parse_pdf()
    vector_db = creating_vectorDB(book_text)
    response = create_retriever(vector_db, query)
    pair_json = create_pair_json(query, response)
    # return pair_json
    return response


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    response = my_function(text)
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()