from gensim.models import Word2Vec

def train_word_embeddings(sentences, embed_size=256, window=5, min_count=1):
    """Train Word2Vec embeddings."""
    model = Word2Vec(sentences, vector_size=embed_size, window=window, min_count=min_count, sg=0)
    return model
