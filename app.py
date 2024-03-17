
import modal
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import vecstore
from utils import pretty_log


image = modal.Image.debian_slim( 
    python_version="3.10"
).pip_install( 
    "langchain",
    "openai",
    "langchain-openai",
    "faiss-gpu",
    "pymongo[srv]",
    "gradio",
    "langchainhub", force_build=True
)


stub = modal.Stub(
    name="emanual-backend",
    image=image,
    secrets=[
        modal.Secret.from_name("mongo-emanual-secret")
    ],
    mounts=[
        modal.Mount.from_local_python_packages(
            "vecstore", "docstore", "utils", "prompts"
        )
    ],
)

VECTOR_DIR = vecstore.VECTOR_DIR
vector_storage = modal.NetworkFileSystem.persisted("vector-vol")


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
@modal.web_endpoint(method="GET")
def web(query: str, request_id=None):

    pretty_log(
        f"handling request with client-provided id: {request_id}"
    ) if request_id else None

    answer = qanda.remote(
        query,
        request_id=request_id
    )
    return {"answer": answer}


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
def qanda(query: str, request_id=None) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain_openai import ChatOpenAI

    import vecstore, prompts

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0, max_tokens=256)
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=prompts.main,
        document_variable_name="sources",
    )

    #pretty_log(f"input_documents: {sources}, question: {query}")

    result = chain.invoke(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )

    answer = result["output_text"]
    #pretty_log(f"answer: {answer}")
    return answer


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    cpu=8.0,  # use more cpu for vector storage creation
)
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")


@stub.function(image=image)
def drop_docs(collection: str = None, db: str = None):
    """Drops a collection from the document storage."""
    import docstore

    docstore.drop(collection, db)


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
def cli(query: str):
    answer = qanda.remote(query)
    pretty_log("ANSWER")
    print(answer)

web_app = FastAPI(docs_url=None)


@web_app.get("/")
async def root():
    return {"message": "See /gradio for the dev UI."}


@web_app.get("/docs", response_class=RedirectResponse, status_code=308)
async def redirect_docs():
    """Redirects to the Gradio subapi docs."""
    return "/gradio/docs"


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
@modal.asgi_app(label="emanual-backend")
def fastapi_app():
    """A simple Gradio interface for debugging."""
    import gradio as gr
    from gradio.routes import App

    def chain_qanda(*args, **kwargs):
        return qanda.remote(*args, **kwargs)

    inputs = gr.TextArea(
        label="Question",
        value="How do I make a coffee?",
        show_label=True,
    )
    outputs = gr.TextArea(
        label="Answer", value="The answer will appear here.", show_label=True
    )

    interface = gr.Interface(
        fn=chain_qanda,
        inputs=inputs,
        outputs=outputs,
        title="Ask Emanual",
        description="I am your friendly helper Emanual. I remember all your household manuals including your car. Also I can speak different languages",
        examples=[
            "Is my Samsung TV equipped with Internet security?",
            "How do I clean windshield?",
            "How do I make cappuccino?",
            "How do you adjust the front seats?",
            "What's the proper way to fasten a seat belt?",
            "How do you install a rear-facing child restraint?",
            "When should seat belt extenders be used?",
            "Why shouldn't the vehicle be modified?",
            "What does the automatic drive positioner do?",
            "How should pregnant women wear a seat belt?",
            "How do you use the seat belt's Automatic Locking Retractor (ALR)?",
            "How do you fill the boiler with water upon first use?",
            "What is the factory setting for the PID temperature for coffee?",
            "How can you manually adjust the brewing pressure?",
            "How do you program the PID to show temperature in °C or °F?",
            "What are the steps to clean the brew group?",
            "How often should the boiler water be changed?",
            "What is the process to descale the machine?",
            "How can you pair the Samsung Smart Remote to the TV?",
            "What should you do if the TV won't turn on?",
            "Where is the Eco Sensor located on the TV?",
            "How do you access the e-Manual on the TV?",
            "What is the purpose of the One Invisible Connection?",
            "How should you clean the TV screen to avoid scratches?",
            "What steps should you follow for the initial setup of the TV?",
            "How can you reduce the power consumption of the TV?",
            "What should you do if the remote control does not work?",
            "How can you secure the TV to the wall to prevent it from falling?",
        ],
        allow_flagging="never",
        theme=gr.themes.Default(radius_size="none", text_size="lg"),
    )

    interface.dev_mode = False
    interface.config = interface.get_config_file()
    interface.validate_queue_settings()
    #interface.allowed_paths = [absolute_path]
    gradio_app = App.create_app(
        interface, app_kwargs={"docs_url": "/docs", "title": "ask-emanual"}
    )

    @web_app.on_event("startup")
    async def start_queue():
        if gradio_app.get_blocks().enable_queue:
            gradio_app.get_blocks().startup_events()

    web_app.mount("/gradio", gradio_app)

    return web_app
