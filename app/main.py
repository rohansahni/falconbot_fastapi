import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from fuzzywuzzy import process
from langdetect import detect
# from deep_translator import GoogleTranslator
# from deep_translator.exceptions import RequestError
import time

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyApOq_jsAbRAaR4O2cyfbx2aGQ_xvBFMSY"

# === FastAPI App ===
app = FastAPI(
    title="E-Command RAG Assistant",
    description="Ask questions from the user manual file using Gemini and LangChain",
    version="1.0"
)

# === Request Body Schema ===
class QueryInput(BaseModel):
    query: str

# === Load Image Mapping from JSON ===
with open("image_mapping.json", "r", encoding="utf-8") as f:
    IMAGE_MAPPING = json.load(f)

def append_images_if_applicable(answer: str, question: str) -> str:
    matched_images = []
    
    for keyword, image_urls in IMAGE_MAPPING.items():
        best_match, score = process.extractOne(question.lower(), [keyword.lower()])
        if score > 80:  # Adjust threshold as needed
        # if keyword.lower() in question.lower():
            for image_url in image_urls:
                markdown_image = f"![{keyword}]({image_url})"
                matched_images.append(markdown_image)
    
    if matched_images:
        return answer + "\n\n" + "\n".join(matched_images)
    
    return answer


loader1 = TextLoader("e-commandcontrol_usermanual_en.txt")
english_docs = loader1.load()


translated_file_path = "arabic_to_english_translated_manual.txt"


if os.path.exists(translated_file_path):
    with open(translated_file_path, "r", encoding="utf-8") as f:
        translated_text = f.read()
# else:

#     with open("e-commandcontrol_usermanual_ar.txt", "r", encoding="utf-8") as f:
#         arabic_text = f.read()


#     splitter = RecursiveCharacterTextSplitter(chunk_size=4800, chunk_overlap=200)
#     chunks = splitter.split_text(arabic_text)


#     def safe_translate(text, retries=3, delay=2):
#         for attempt in range(retries):
#             try:
#                 return GoogleTranslator(source='ar', target='en').translate(text)
#             except RequestError:
#                 if attempt < retries - 1:
#                     print(f"Translation failed, retrying in {delay}s...")
#                     time.sleep(delay)
#                 else:
#                     raise

#     # === Translate chunks ===
#     translated_chunks = []
#     for chunk in chunks:
#         translated_chunk = safe_translate(chunk)
#         translated_chunks.append(translated_chunk)

#     translated_text = "\n\n".join(translated_chunks)

#     # === Save translation to file for future use ===
#     with open(translated_file_path, "w", encoding="utf-8") as f:
#         f.write(translated_text)


loader2 = TextLoader("arabic_to_english_translated_manual.txt")
translated_docs = loader2.load()


documents = english_docs + translated_docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
 
docs = text_splitter.split_documents(documents)

# === Embeddings ===
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# === Vector DB ===
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="chroma_db"
)
vectordb.persist()

# === Gemini Chat Model ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# === RetrievalQA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# === Routes ===
@app.get("/")
def read_root():
    return {"message": "🚀 E-Command RAG Assistant is running!"}

@app.post("/ask/")
def ask_question(query_input: QueryInput):
    user_query = query_input.query

    # Detect query language
    user_lang = detect(user_query)

    translated_query = GoogleTranslator(source=user_lang, target="en").translate(user_query)
    print(translated_query)
    
    # Retrieve answer from English manual
    result = qa_chain({"query": translated_query})
    print(result)
    
    # Translate English answer into detected language
    translated_answer = GoogleTranslator(source="en", target=user_lang).translate(result["result"])
    print(translated_answer)

    # Append images
    answer_with_image = append_images_if_applicable(translated_answer, user_query)

    print(answer_with_image)
    
    return {
        "question": user_query,
        "answer": answer_with_image
    }
