from fastapi import FastAPI, UploadFile, HTTPException
from typing import List
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
import nest_asyncio

# Apply nest_asyncio at the start
nest_asyncio.apply()

load_dotenv()

app = FastAPI()

# Initialize global variables
parser = LlamaParse(
    api_key=os.getenv("LLAMA_KEY"),
    result_type="markdown",
)

llm = OpenAI(model="gpt-4o-mini")
node_parser = MarkdownElementNodeParser(llm=llm, num_workers=4)
documents = []
index = None


@app.post("/documents/")
async def add_document(file: UploadFile):
    global documents, index

    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Parse document - using aload_data instead of load_data
        new_documents = await parser.aload_data(temp_path)
        documents.extend(new_documents)

        # Update index
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        index = await VectorStoreIndex.acreate(nodes=base_nodes + objects, llm=llm)

        # Cleanup
        os.remove(temp_path)

        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/")
async def delete_documents():
    global documents, index
    documents = []
    index = None
    return {"message": "All documents deleted"}


@app.post("/query/")
async def query_documents(query: str):
    if not index:
        raise HTTPException(status_code=400, detail="No documents have been indexed")

    query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
    response = await query_engine.aquery(query)

    return {
        "response": str(response),
        "sources": [node.get_content() for node in response.source_nodes],
    }
