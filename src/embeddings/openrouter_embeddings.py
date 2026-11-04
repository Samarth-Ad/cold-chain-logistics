import os
import requests

from langchain_core.embeddings import Embeddings


class OpenRouterNemotronEmbeddings(Embeddings):

    def __init__(self):
        self.api_key = os.getenv("OPEN_ROUTER_API_KEY")
        self.base_url = os.getenv(
            "OPEN_ROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        )
        self.model = os.getenv(
            "OPEN_ROUTER_EMBEDDING_MODEL_NAME",
            "nvidia/nemotron-3-embed-1b:free"
        )

        if not self.api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is missing")

        self.session = requests.Session()

        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:

        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": texts,
                "input_type": input_type,
                "encoding_format": "float"
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()["data"]

        # Preserve original input order
        data.sort(key=lambda item: item["index"])

        return [
            item["embedding"]
            for item in data
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        return self._embed(
            texts=texts,
            input_type="passage"
        )

    def embed_query(self, text: str) -> list[float]:

        return self._embed(
            texts=[text],
            input_type="query"
        )[0]