import llama_cpp
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

import instructor

from pydantic import BaseModel
from rich.console import Console
from enum import Enum


# Initialisation du modèle Llama
llm = llama_cpp.Llama(
    model_path="/home/yassir/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=2000,
    n_batch=2000,
    main_gpu=0,
    n_gpu_layers=-1,
    Temperature=0,
    top_k=1,
    verbose=True,
    seed=69,
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2)
)

# Patch de la méthode de création
create = instructor.patch(
    create=llm.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA  # (2)!
)

# Définition des classes pour les informations à extraire
class ArticleType(str, Enum):
    biaisé = "biaisé"
    impartial = "impartial"

class ArticleInfo(BaseModel):
    biaisé_ou_impartial: ArticleType
    justification: str

    class Config:
        use_enum_values = True

# Texte de l'article à analyser
article_text = """
"""

# Création du flux d'extraction
extraction_stream = create(
    response_model=instructor.Partial[ArticleInfo],  # (3)!
    messages=[
        {
            "role": "system",
            "content": """Vous êtes un assistant spécialisé dans l'analyse des articles de presse de langue française. Votre tâche est de déterminer si un article est biaisé ou impartial. Identifiez les indices de partialité potentielle, tels que des éloges excessifs ou des mentions répétitives d'une marque, qui pourraient suggérer que l'article est sponsorisé. Fournissez également une évaluation globale de l'objectivité de l'article. Vous devez répondre par si la news est "biaisé" ou "impartial", puis justifier votre réponses en français."""
        },
        {
            "role": "user",
            "content": article_text
        },
    ],
    stream=True,
)

console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()  # (4)!
    console.print(obj)
