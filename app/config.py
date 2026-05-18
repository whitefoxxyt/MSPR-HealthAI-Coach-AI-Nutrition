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
    better_auth_secret: str = ""
    auth_api_url: str = "http://mspr-healthai-auth:3000"
    # Toggle few-shot prompting (slice 9). Mis a false uniquement pour l'eval
    # comparative "Impact du few-shot" dans docs/metrics.md.
    few_shot_enabled: bool = True

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
