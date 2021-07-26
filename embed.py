from transformers import AutoTokenizer, AutoModel
import json, torch, torch.nn as nn, seaborn as sns, matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

classes = json.load(open('AVE_Dataset/classes.json'))

# BERT embeddings 
embeddings, similarities = torch.zeros([28, 768]), torch.zeros([28, 28])
for i, name in enumerate(classes.keys(), 0):
    inputs = tokenizer(name, return_tensors="pt")
    embeddings[i] = model(**inputs)['pooler_output'][0]

# calculate similarities
threshold = 2
cos = nn.CosineSimilarity(dim=0)
for i in range(embeddings.size(0)):
    for j in range(embeddings.size(0)):
        current_sim = cos(embeddings[i], embeddings[j])
        if current_sim < threshold and i >= j:
            similarities[i][j] = current_sim

# plot similarities
plt.subplots(figsize=(20,15))
plot = sns.heatmap(data=similarities.detach().numpy(), 
                   xticklabels=classes.keys(), 
                   yticklabels=classes.keys(), 
                   cmap="YlGnBu")
figure = plot.get_figure()
figure.savefig("cosine.jpg")