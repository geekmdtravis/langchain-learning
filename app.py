from flask import Flask, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from transformers import GPT2TokenizerFast
import os

HF_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
    temperature=0.001,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    huggingfacehub_api_token=HF_API_TOKEN,
)




template="""
You are a cryptocurrency policy export. You will answer
the question below using the relevant information from the
similar chunks of documents provided.

Question: {question}

Relevant Information: {similar_chunks}

Answer:"""


app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return """
    <html>
        <body>
            <h1>LlaMeR GPT</h1>
            <button>Click me! (Please)</button>
        </body>
        <p>
            At the "generate/" endpoint, you can generate text using the LlaMeR GPT model.
            Please provide the query parameters "video" and "query" to generate text. 
            Video is the YouTube video ID and query is the text you want to generate.
        </p>
    </html>
    """


@app.route("/generate", methods=["GET"])
def generate():
    question = request.args.get("query", default="", type=str)
    video = request.args.get("video", default="", type=str)
    SOURCE = f"https://youtu.be/{video}"
    loader = YoutubeLoader.from_youtube_url(SOURCE, add_video_info=False)
    docs = loader.load()
    char_count = len(docs[0].page_content)
    print(f"Char count in {SOURCE}: {char_count}")

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    cts = RecursiveCharacterTextSplitter(separators=["|\n\n", "\n", " ", ""])
    splitter = cts.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=100, chunk_overlap=10
    )
    split_docs = splitter.split_documents(docs)
    print(f"Split document count: {len(split_docs)}")

    for index, doc in enumerate(split_docs):
        doc.metadata.update({"chunk_id": index})
        doc.metadata.update({"token_count": len(tokenizer.tokenize(doc.page_content))})


    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_TOKEN, model_name="sentence-transformers/all-mpnet-base-v2"
    )

    doc_content = [doc.page_content for doc in split_docs]
    vectorized_docs = embeddings.embed_documents(doc_content)

    query = "Make a detailed summary of the youtube video conversation."
    query_vector = embeddings.embed_query(query)

    db = FAISS.from_documents(split_docs, embeddings)
    found = db.similarity_search_by_vector(query_vector, k=30)
    sorted_found = sorted(found, key=lambda x: x.metadata["chunk_id"])
    prompt = PromptTemplate(template=template, input_variables=["question", "similar_chunks"])

    chain: Runnable = prompt | llm

    res = chain.invoke({"question": question, "similar_chunks": [doc.page_content for doc in sorted_found]})
    clean_res = res.replace("<|eot_id|>", " — Hal 9000")
    return f"<html><body><h1>Generated Text</h1><p>{clean_res}</p></body></html>"


if __name__ == "__main__":
    app.run(debug=True)
