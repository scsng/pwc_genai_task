


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        return self.model.embed(text)