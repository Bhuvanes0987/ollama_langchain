# Ollama
This repo brings numerous use cases from the Open Source Ollama

# Step 1
Go Ahead to https://ollama.ai/ and download the set up file

# Step 2
Install and Start the Software

# Step 3
The Repo has numerous working case as separate Folders. You can work on any folder for testing various use cases



https://github.com/PromptEngineer48/Ollama/tree/main

ttps://github.com/PromptEngineer48/Ollama/tree/main/2-ollama-privateGPT-chat-with-docs


to clone

https://github.com/PromptEngineer48/Ollama.git


pip install -r requirements.txt

ollama pull mistral

mkdir source_documents

python ingest.py

Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.99s/it]
Loaded 235 new documents from source_documents
Split into 1268 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Ingestion complete! You can now run privateGPT.py to query your documents

python privateGPT.py

ollama pull llama2:13b
MODEL=llama2:13b python privateGPT.py

.csv: CSV,
.docx: Word Document,
.doc: Word Document,
.enex: EverNote,
.eml: Email,
.epub: EPub,
.html: HTML File,
.md: Markdown,
.msg: Outlook Message,
.odt: Open Document Text,
.pdf: Portable Document Format (PDF),
.pptx : PowerPoint Document,
.ppt : PowerPoint Document,
.txt: Text file (UTF-8),




hugging face error

pip install --upgrade huggingface_hub


pip install --upgrade sentence_transformers










#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "mistral")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()






#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time

# Initialize environment variables
model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

# Function to remove duplicates from the Chroma database
def remove_duplicates_from_db(db, new_documents):
    """
    Remove documents that are already in the database by comparing content.
    """
    existing_docs = db.get_documents()  # Assuming db.get_documents() returns a list of stored documents
    existing_texts = [doc.page_content for doc in existing_docs]
    
    # Filter new documents to remove those already in the database
    unique_documents = [doc for doc in new_documents if doc.page_content not in existing_texts]
    
    return unique_documents

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Example: Assume we load documents from a source (this can be changed to your source)
    new_documents = [
        "This is a sample document.",
        "This is another document.",
        "This is a sample document.",  # Duplicate
        "Another unique document."
    ]
    
    # Remove duplicates from the new documents
    unique_documents = remove_duplicates_from_db(db, new_documents)

    # Add the unique documents to Chroma if there are any
    if unique_documents:
        db.add_texts([doc.page_content for doc in unique_documents])

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()


sentence transformers error

pip install --upgrade sentence_transformers

pip install --upgrade huggingface_hub

pip install huggingface_hub==0.12.0

pip list

pip install --upgrade <package-name>

# Create a new virtual environment
conda create --name new_env python=3.8
conda activate new_env

# Install necessary packages
pip install sentence_transformers langchain huggingface_hub

to rename 

 Rename-Item -Path "1-ollama-langchain" -NewName "ollama_langchain"
>>
(base) PS C:\Users\LENOVO\Documents\chat_with_docs\Ollama> Rename-Item -Path "1-ollama-langchain" -NewName "ollama_langchain"
