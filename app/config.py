from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "mspr-healthai-db"
    db_port: int = 5432
    db_name: str = "healthai"
    db_user: str = "healthai_user"
    db_password: str = "password"
    ollama_host: str = "http://ollama:11434"
    better_auth_secret: str = ""

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
