import torch
from transformers import AutoModel
from retrieval.config import (
    MODEL_NAME,
    MODEL_REVISION,
    MAX_TOKENS_BY_MODALITY,
    MAX_INPUT_TILES,
    USE_THUMBNAIL,
)


def load_embedding_model(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> AutoModel:
    """Load the multimodal embedding model onto the best available device.

    The model is loaded in float16 with Flash Attention 2 and placed into
    eval mode so no gradients are tracked.

    Args:
        model_name: HuggingFace model identifier.
        revision:   Specific git commit SHA for reproducibility.

    Returns:
        Loaded model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading embedding model '{model_name}' on {device} …")

    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()

    print("Model loaded successfully.")
    return model


def prepare_processor(modality: str, embedding_model: AutoModel):
    """Set the processor's token limits for the requested modality.

    The model's processor must be reconfigured before encoding because
    text, image, and combined queries have different token budgets.

    Args:
        modality:        One of ``"text"``, ``"image"``, or ``"image_text"``.
        embedding_model: The already-loaded embedding model.

    Returns:
        Tuple ``(modality, embedding_model)`` — returned for convenience so
        callers can write ``modality, model = prepare_processor(...)``.

    Raises:
        ValueError: If ``modality`` is not one of the supported values.
    """
    if modality not in MAX_TOKENS_BY_MODALITY:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Choose from {list(MAX_TOKENS_BY_MODALITY)}."
        )

    embedding_model.processor.p_max_length  = MAX_TOKENS_BY_MODALITY[modality]
    embedding_model.processor.max_input_tiles = MAX_INPUT_TILES
    embedding_model.processor.use_thumbnail   = USE_THUMBNAIL

    return modality, embedding_model