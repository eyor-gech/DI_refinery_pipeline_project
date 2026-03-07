import numpy as np
from fastembed import TextEmbedding


class SimpleVectorStore:

    def __init__(self):
        # FastEmbed model (downloads automatically on first run)
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.vectors = []
        self.payloads = []

    def add(self, content, payload):

        # fastembed returns a generator → convert to list
        emb = list(self.model.embed([content]))[0]

        self.vectors.append(np.array(emb))
        self.payloads.append(payload)

    def search(self, query, top_k=3):

        q = np.array(list(self.model.embed([query]))[0])

        sims = []
        for i, v in enumerate(self.vectors):

            score = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))

            sims.append((score, i))

        sims.sort(reverse=True)

        return [
            {**self.payloads[i], "score": score}
            for score, i in sims[:top_k]
        ]