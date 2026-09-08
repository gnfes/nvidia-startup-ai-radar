"""The 24 sources listed in the TAPI brief, section 8 (Fontes para base de
conhecimento NVIDIA). Static list — the brief names these explicitly, so
there's no discovery step like ingestion/sources.py has for startups.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KbSource:
    categoria: str  # 'material_apoio' | 'documentacao_oficial'
    tipo: str  # 'artigo' | 'blog' | 'documentacao' | 'video'
    titulo: str
    url: str


# 8.1 Materiais de apoio do case
MATERIAIS_APOIO: list[KbSource] = [
    KbSource("material_apoio", "artigo", "Sequoia — Services: The New Software", "https://sequoiacap.com/article/services-the-new-software/"),
    KbSource("material_apoio", "artigo", "Emergence Capital — The AI-Native Services Playbook", "https://www.emcap.com/thoughts/the-ai-native-services-playbook"),
    KbSource("material_apoio", "blog", "NVIDIA AI 5-Layer Cake", "https://blogs.nvidia.com/blog/ai-5-layer-cake/"),
    KbSource("material_apoio", "video", "Playlist de tecnologias NVIDIA", "https://youtube.com/playlist?list=PLBaUJRFQ-j_WJZdZfFNsgUWDWF1Ldjp_X"),
    KbSource("material_apoio", "video", "Comunidade de startups NVIDIA", "https://youtu.be/NmZDQSdUVUQ"),
    KbSource("material_apoio", "video", "Benefícios do NVIDIA Inception", "https://www.youtube.com/live/fWfkE6cibwQ"),
]

# 8.2 Documentações oficiais NVIDIA
DOCUMENTACOES_OFICIAIS: list[KbSource] = [
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Inception", "https://www.nvidia.com/en-us/startups/"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA NIM", "https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA API Catalog", "https://build.nvidia.com/"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA NeMo", "https://www.nvidia.com/en-us/ai-data-science/products/nemo/"),
    KbSource("documentacao_oficial", "documentacao", "NeMo Guardrails", "https://github.com/NVIDIA/NeMo-Guardrails"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Triton Inference Server", "https://developer.nvidia.com/triton-inference-server"),
    KbSource("documentacao_oficial", "documentacao", "Triton Inference Server — Docs", "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/"),
    KbSource("documentacao_oficial", "documentacao", "TensorRT-LLM", "https://github.com/NVIDIA/TensorRT-LLM"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA RAPIDS", "https://rapids.ai/"),
    KbSource("documentacao_oficial", "documentacao", "cuDF", "https://docs.rapids.ai/api/cudf/stable/"),
    KbSource("documentacao_oficial", "documentacao", "cuML", "https://docs.rapids.ai/api/cuml/stable/"),
    KbSource("documentacao_oficial", "documentacao", "CUDA Toolkit", "https://developer.nvidia.com/cuda-toolkit"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Riva", "https://developer.nvidia.com/riva"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Omniverse", "https://www.nvidia.com/en-us/omniverse/"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Isaac", "https://developer.nvidia.com/isaac"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Clara", "https://www.nvidia.com/en-us/clara/"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA Morpheus", "https://developer.nvidia.com/morpheus-cybersecurity"),
    KbSource("documentacao_oficial", "documentacao", "NVIDIA AI Enterprise", "https://www.nvidia.com/en-us/data-center/products/ai-enterprise/"),
]

ALL_SOURCES: list[KbSource] = MATERIAIS_APOIO + DOCUMENTACOES_OFICIAIS
