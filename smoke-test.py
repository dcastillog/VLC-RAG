from fastembed import TextEmbedding
m = TextEmbedding("BAAI/bge-small-en-v1.5")
v = list(m.embed(["DCO-OFDM in visible light communication"]))[0]
print(len(v))