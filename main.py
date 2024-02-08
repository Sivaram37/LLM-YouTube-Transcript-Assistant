from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import textwrap

# Set the OPENAI_API_KEY environment variable or replace "your-api-key" with your actual API key
openai_api_key = "your key"
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

video_url = "https://www.youtube.com/watch?v=NYSWn1ipbgg&list=PL-Y17yukoyy3zzoMJNkWQuogKbWGyBL-d"

def create_db_from_youtube_video(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    chunk_size = 100
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap= chunk_overlap)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    template = """
    You are a helpful assistant that can answer questions about
    Youtube videos based on the vidoes's transcript: {docs}
    Only use the factual information from the transcript to answer the 
    question.
    If you feel like you don't have enough information to answer
    the question, say "I don't Know".
     """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, prompt=chat_prompt)
    response = response.replace("\n", "")
    return response, docs

# Example usage
db = create_db_from_youtube_video(video_url)
query = "Your question here"
response, docs = get_response_from_query(db, query)
print("Response:", response)
