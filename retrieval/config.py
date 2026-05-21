# Qdrant connection
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
COLLECTION_NAME: str = "image_text_index"

MODEL_NAME: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
MODEL_REVISION: str = "062ffaa1e3d24a8a50bd6a7ac7b8e54103e1f01d"

MAX_TOKENS_BY_MODALITY: dict[str, int] = {
    "image":      2_048,
    "image_text": 10_240,
    "text":       8_192,
}

# Image tiling settings
MAX_INPUT_TILES: int = 6     
USE_THUMBNAIL: bool = True   

FIGURE_HEIGHT: int = 7          
AXES_PER_RESULT_WIDTH: int = 6  
MARKDOWN_SNIPPET_CHARS: int = 500
MARKDOWN_WRAP_WIDTH: int = 42
MARKDOWN_FONT_SIZE: int = 7