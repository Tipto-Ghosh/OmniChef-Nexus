import warnings
warnings.filterwarnings("ignore")
import os
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache"

from datasets import load_dataset
from retrieval.model_loader import load_embedding_model
from retrieval.db_client import get_qdrant_client
from retrieval.visualizer import display_results
from reranker import load_rerank_model, retrieve_and_rerank, compare_reranking_results
from transformers.image_utils import load_image

embedding_model = load_embedding_model()
rerank_model = load_rerank_model()

# connect to Qdrant
qdrant_client = get_qdrant_client()

# load dataset 
dataset = load_dataset("tiptoghosh/food-recipes-15k", split="train")
print("Ready For quering......")


text_query  = "give me a salad recipe that looks like the given image"
image_query = load_image("sample/example_ingredients.jpg")

output = retrieve_and_rerank(
    embedding_model = embedding_model,
    qdrant_client = qdrant_client,
    rerank_model = rerank_model,
    dataset = dataset,
    query_text  = text_query,
    query_image = image_query,
    retrieval_top_k = 20,
    final_top_n = 5,
)

compare_reranking_results(output["retrieved"], output["reranked"])
display_results(output["top_n"], query_label=f"[Text+Image] {text_query}")