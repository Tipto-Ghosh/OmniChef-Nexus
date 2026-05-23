RERANK_MODEL_NAME = "nvidia/llama-nemotron-rerank-vl-1b-v2"
PROCESSOR_BASE_KWARGS = {
    "trust_remote_code": True,
    "max_input_tiles": 6,
    "use_thumbnail": True
}

RERANK_MAX_TOKENS_BY_MODALITY = {
    "image": 2048,
    "text": 8192,
    "image_text": 10240,
}

DOC_TEXT_COLUMN = "markdown"
DOC_IMAGE_COLUMN = "image"