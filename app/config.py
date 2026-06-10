from typing import Literal

from pydantic_settings import BaseSettings

# Reperes ANSES (rapport "Actualisation des reperes du PNNS", 2016 ;
# avis 2017-SA-0142 sur les recommandations alimentaires).
# Utilises par nutrition_engine et imbalance_detector pour calculer les
# desequilibres macro a partir d'une cible personnalisee (TDEE) plutot que
# de seuils en pourcentage figes.
RNP_PROTEIN_G_PER_KG = 0.83  # apport recommande, en g/kg de poids corporel/j
RNP_FIBER_G_PER_DAY = 30  # cible journaliere fibres alimentaires, en g/j
RNP_AGS_PERCENT_OF_AET_MAX = 0.12  # plafond acides gras satures, en % AET
RNP_TOTAL_SUGARS_G_MAX = 100  # plafond sucres totaux, en g/j


class Settings(BaseSettings):
    db_host: str = "mspr-healthai-db"
    db_port: int = 5432
    db_name: str = "healthai"
    db_user: str = "healthai_user"
    db_password: str = "password"
    ollama_host: str = "http://ollama:11434"
    # Options d'inference Ollama. num_ctx explicite : le defaut Ollama
    # (2048-4096 selon version) tronque silencieusement le prompt few-shot
    # (~2245 tokens), ce qui degrade la coherence des plans. Temperature
    # basse : les sorties sont contraintes par schema JSON, on privilegie
    # le determinisme.
    ollama_model: str = "gemma3:4b"
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.2
    better_auth_secret: str = ""
    auth_api_url: str = "http://mspr-healthai-auth:3000"
    # Toggle few-shot prompting (slice 9). Mis a false uniquement pour l'eval
    # comparative "Impact du few-shot" dans docs/metrics.md.
    few_shot_enabled: bool = True
    # Exemples few-shot complets (7j/5j/1j) au lieu du 1er jour seul. A
    # reserver aux infra GPU : le prefill CPU de Gemma3:4b timeout au-dela
    # de ~2000 tokens (cf. docs/GPU_EVAL_PLAYBOOK.md).
    few_shot_full_examples: bool = False
    # Defaut LLM (PRD #71). Renomme de llm_backend en default_llm au slice 3
    # pour refleter qu'il s'agit d'une valeur de repli quand l'utilisateur
    # n'a pas exprime de preference (NutritionGoal.preferred_llm). L'env var
    # DEFAULT_LLM remplace LLM_BACKEND ; voir .env.example.
    # Literal pour fail-fast au boot si l'env est mal configure : sans ca une
    # valeur invalide ne casserait qu'au runtime via LLMBackend(...) cote
    # user_preferences_service / llm_provider.
    default_llm: Literal["ollama", "mistral"] = "mistral"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    # Backend de classification photo (/analyze-meal). "food101" = classifieur
    # HuggingFace local (defaut historique valide). "mistral_vision" = vision
    # Mistral contrainte au catalogue Food-101 (meilleure reconnaissance,
    # multi-aliments), avec repli automatique sur food101 en cas d'echec.
    analyze_backend: Literal["food101", "mistral_vision"] = "food101"
    mistral_vision_model: str = "pixtral-12b-2409"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
