import pickle as pkl
import numpy as np
import json
import pdb
from tqdm import tqdm

K = 20

#article_info = json.load(open('../data/articles.json'))
#articles = article_info.keys()
#embed = json.load(open('../data/glove_embed.json'))
#
#mat = {}
#
#for art in tqdm(articles):
#    text = article_info[art]['title'] + article_info[art]['text']
#    temp = []
#    for word in text[:K]:
#        if word.lower() in embed:
#            temp.append(embed[word.lower()])
#        else:
#            temp.append([0]*300)
#    for i in range(K-len(text)):
#            temp.append([0]*300)
#
#    mat[art] = np.asarray(temp)
#
#pkl.dump(mat, open('../data/mat.pkl', 'w'))

mat = pkl.load(open('../data/mat.pkl'))

user_hist = json.load(open('../data/user_history.json'))
users = user_hist.keys()

user_mat = {}

for user in tqdm(users):
    temp = []
    for art in user_hist[user]:
        temp.append(mat[str(art)])
    user_mat[user] = np.asarray(temp)

pdb.set_trace()
pkl.dump(user_mat, open('../data/user_mat.pkl', 'w'))
