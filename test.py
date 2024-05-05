# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

# # prepare input
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')

# # forward pass
# output = model(**encoded_input)
# print(output)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

# Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

print(embedding)