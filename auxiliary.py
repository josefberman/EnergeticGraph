from sentence_transformers import SentenceTransformer, models
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import logging as hf_logging
import logging

hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Determine appropriate device for embeddings (CPU fallback if CUDA unavailable)
try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"

def ChemBERT_ChEMBL_pretrained():
    # 2. load the raw HuggingFace ChemBERT model as a Transformer module
    word_embedding_model = models.Transformer(
        model_name_or_path="jonghyunlee/ChemBERT_ChEMBL_pretrained",
        max_seq_length=1024  # adjust as needed
    )

    # 3. add a mean-pooling layer on top of the token embeddings
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    # 4. assemble your SentenceTransformer
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def ChemBERT_ChEMBL_pretrained_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jonghyunlee/ChemBERT_ChEMBL_pretrained",
        model_kwargs={"device": _DEVICE}
    )

def allenai_specter_pretrained_embeddings():
    return HuggingFaceEmbeddings(
        model_name="allenai/specter",
        model_kwargs={"device": _DEVICE}
    )

def all_mini_l6_v2_pretrained_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": _DEVICE}
    )
