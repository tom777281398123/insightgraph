"""Flask entry point for the knowledge-graph algorithm service.

The development configuration uses an OpenAI-compatible LLM API.  Local LLM
loading remains available for deployments that explicitly set ``LLM_MODE = 0``.
"""

import os

from flask import Flask, g
from flask_executor import Executor
from sentence_transformers import SentenceTransformer

from log.log import logging
from value.global_value import LLM_BASE_URL, LLM_MODE, LLM_MODEL_NAME


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = (
    "/Users/SHIMMER/szh/大模型开发/Agent/美团（2606-2610）/项目/"
    "storekeeper-nl2sql/docker/models/bge-large-zh-v1.5"
)

os.chdir(PROJECT_DIR)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False
app.config["EXECUTOR_TYPE"] = "thread"
app.config["EXECUTOR_MAX_WORKERS"] = 1
executor = Executor(app)

logging.info("Current working directory: %s", os.getcwd())

model_loader = None
if LLM_MODE == 0:
    # ``data.classes`` imports vLLM, so defer it unless local inference is used.
    from data.classes import ModelLoader

    logging.info("Loading local LLM model...")
    model_loader = ModelLoader()
    tokenizer, model = model_loader.load_model(4)
elif LLM_MODE == 1:
    logging.info("Using OpenAI-compatible LLM API: %s", LLM_BASE_URL)
    tokenizer, model = LLM_BASE_URL, LLM_MODEL_NAME
else:
    raise ValueError(f"Unsupported LLM_MODE: {LLM_MODE}")

logging.info("Loading embedding model from %s", EMBEDDING_MODEL_PATH)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
logging.info("Embedding model loaded.")


@app.before_request
def load_request_resources():
    """Expose the shared models and executor to existing Blueprint handlers."""
    g.model_loader = model_loader
    g.tokenizer = tokenizer
    g.model = model
    g.executor = executor
    g.embedding_model = embedding_model


# Keep the original Blueprint registration order for endpoint compatibility.
from api.test.api_hello import blueprint_hello
from api.file_stream.api_file_stream import blueprint_image_up_down
from api.knowledge_extraction.api_event_extraction import blueprint_event_extraction
from api.knowledge_extraction.api_knowledge_extraction import blueprint_entity_extraction
from api.knowledge_extraction.api_knowledge_fusion import blueprint_KnowledgeFusion
from api.graph_fusion.api_graph_fusion import blueprint_graphFusion
from api.kg_clustering.api_clustering import blueprint_kg_cluster
from api.event_analysis.api_event_prediction import blueprint_event_prediction
from api.event_analysis.api_event_causation import blueprint_event_causation
from api.error_correct.api_error_correct import blueprint_error_correct
from api.if_then.api_if_then import blueprint_if_then
from api.object_analysis.api_object_analysis import blueprint_object_analysis
from api.training_model.api_training_model import blueprint_training_model

for blueprint in (
    blueprint_hello,
    blueprint_image_up_down,
    blueprint_event_extraction,
    blueprint_entity_extraction,
    blueprint_KnowledgeFusion,
    blueprint_graphFusion,
    blueprint_kg_cluster,
    blueprint_event_prediction,
    blueprint_event_causation,
    blueprint_error_correct,
    blueprint_if_then,
    blueprint_object_analysis,
    blueprint_training_model,
):
    app.register_blueprint(blueprint)


if __name__ == "__main__":
    logging.info("Starting knowledge-map0318 service...")
    app.run(host="0.0.0.0", port=6004, debug=False)
