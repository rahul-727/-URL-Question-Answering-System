import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import chromadb
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Initialize Hugging Face model for embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Initialize Hugging Face model for text generation
generation_model_name = "sshleifer/distilbart-cnn-12-6"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)

# Initialize ChromaDB
chroma_client = chromadb.Client()
vector_collection = chroma_client.create_collection("url_embeddings")

# Pydantic models
class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Function to extract text from a URL (Depth of 1)
def extract_text_from_url(url: str):
    logger.info(f"Fetching content from URL: {url}")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract main text from the page
    main_text = soup.get_text()
    
    # Extract text from all links (Depth of 1)
    for link in soup.find_all('a', href=True):
        try:
            link_url = link['href']
            # Handle relative URLs
            if not link_url.startswith('http'):
                link_url = requests.compat.urljoin(url, link_url)
            response = requests.get(link_url, timeout=5)
            response.raise_for_status()
            link_soup = BeautifulSoup(response.content, "html.parser")
            link_text = link_soup.get_text(separator=' ', strip=True)
            main_text += " " + link_text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch link: {link['href']} - {e}")

    # Clean the text by removing excess whitespace and newlines
    cleaned_text = ' '.join(main_text.split())
    logger.info("Content fetched successfully.")
    return cleaned_text

# Function to chunk text
def chunk_text(text: str, chunk_size: int = 500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to generate embeddings
def generate_embeddings(text_chunks):
    inputs = embedding_tokenizer(text_chunks, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Function to generate detailed answer using text generation model
def generate_detailed_answer(context: str, query: str, max_length: int = 150):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = generation_tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = generation_model.generate(
        inputs,
        max_length=max_length,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    answer = generation_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer

# Endpoint for URL parsing
@app.post("/url-parser")
async def url_parser(url_request: URLRequest):
    try:
        # Extract text from URL
        main_text = extract_text_from_url(url_request.url)

        # Chunk the text
        text_chunks = chunk_text(main_text)

        # Generate embeddings
        embeddings = generate_embeddings(text_chunks)

        # Store embeddings in ChromaDB
        for chunk, embedding in zip(text_chunks, embeddings):
            # Generate a unique ID for each chunk
            unique_id = str(uuid.uuid4())
            vector_collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[unique_id]
            )

        return {"message": "Embeddings stored successfully."}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=400, detail="Invalid URL or unable to fetch content.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Endpoint for querying the stored embeddings
@app.post("/query")
async def query(query_request: QueryRequest):
    # Validate the incoming query
    if not query_request.query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        # Generate embeddings for the query
        query_embedding = generate_embeddings([query_request.query])[0].tolist()

        # Perform a search against saved embeddings
        results = vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=1  # Fetch top 1 result
        )
        most_relevant_chunks = results['documents'][0] if results['documents'] else []

        if most_relevant_chunks:
            # Combine the chunks if necessary
            context = ' '.join(most_relevant_chunks)
            
            # Generate a detailed answer based on the context
            generated_answer = generate_detailed_answer(context, query_request.query)
            
            return {"answer": generated_answer}
        else:
            return {"message": "No relevant data found."}
    except Exception as e:
        logger.error(f"Error during query processing: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
