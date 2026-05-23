import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor

from reranker.config import RERANK_MODEL_NAME, PROCESSOR_BASE_KWARGS, RERANK_MAX_TOKENS_BY_MODALITY

def load_rerank_model(model_name: str = RERANK_MODEL_NAME) -> AutoModelForSequenceClassification:
    """Load the cross-encoder reranker model onto the best available device.
 
    The model is loaded in float16 with Flash Attention 2 and placed into
    eval mode so no gradients are tracked during inference.
 
    Args:
        model_name: HuggingFace model identifier for the reranker.
 
    Returns:
        Loaded reranker model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading reranker model '{model_name}' on {device} …")
 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
 
    print("Reranker model loaded successfully.")
    return model

def build_rerank_processor(
    modality: str,
    model_name: str = RERANK_MODEL_NAME,
) -> AutoProcessor:
    """Build a reranker processor configured for the given query modality.
 
    The processor's ``rerank_max_length`` must be set at construction time
    (unlike the embedding model's processor which can be patched in-place),
    so this function must be called again whenever the modality changes.
 
    Args:
        modality:   One of ``"text"``, ``"image"``, or ``"image_text"``.
        model_name: HuggingFace model identifier for the reranker processor.
 
    Returns:
        Configured ``AutoProcessor`` ready for cross-encoder scoring.
 
    Raises:
        ValueError: If ``modality`` is not one of the supported values.
    """
    if modality not in RERANK_MAX_TOKENS_BY_MODALITY:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Choose from {list(RERANK_MAX_TOKENS_BY_MODALITY)}."
        )
 
    kwargs = {
        **PROCESSOR_BASE_KWARGS,
        "rerank_max_length": RERANK_MAX_TOKENS_BY_MODALITY[modality],
    }
 
    print(
        f"Building reranker processor for modality='{modality}' "
        f"(max_tokens={kwargs['rerank_max_length']}) …"
    )
 
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    return processor