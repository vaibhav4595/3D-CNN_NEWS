from __future__ import division
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D
from keras.layers.merge import dot, multiply
from keras.callbacks import ModelCheckpoint
import os
import pdb 
import json
import pickle as pkl 
import numpy as np
from tqdm import tqdm
import random

class CNN():

    def __init__(self, user_hist, mat, read_hist, words):

        self.user_hist = json.load(open(user_hist))
        self.users = self.user_hist.keys()
        self.mat = pkl.load(open(mat))
        self.articles = self.mat.keys()
	self.read_hist = read_hist
        self.words = words
	self.model = None

    def create_model(self):

        user_input = Input(shape=(1, self.read_hist, self.words, 300))
        article_input = Input(shape=(1,self.words, 300))

        conv3d = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(user_input)
        maxpooling3d = MaxPooling3D(pool_size=(2, 2, 2))(conv3d)
        conv3d2 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(maxpooling3d)
        maxpooling3d2 = MaxPooling3D(pool_size=(1, 2, 2))(conv3d2)

        flatten3d = Flatten()(maxpooling3d2)
        dense3d = Dense(128, activation='relu')(flatten3d)
        #dropout3d = Dropout(0.3)(dense3d)

        conv2d = Conv2D(64, kernel_size=(3, 3), activation='relu')(article_input)
        maxpooling2d = MaxPooling2D(pool_size=(2, 2))(conv2d)
        conv2d2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxpooling2d)
        maxpooling2d2 = MaxPooling2D(pool_size=(2, 2))(conv2d2)
        flatten2d = Flatten()(maxpooling2d2)
        dense2d = Dense(128, activation='relu')(flatten2d)
        #dropout2d = Dropout(0.5)(dense2d)

        output1 = multiply([dense3d, dense2d])
        output2 = Dense(1, activation='sigmoid')(output1)

        self.model = Model(inputs=[user_input, article_input], outputs = output2)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


    def fit_model(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1)

    def get_model_summary(self):
        print self.model.summary()

    def test_fit_model(self, inputs, output):
        self.model.fit(inputs, output, epochs=50, verbose=1)

def train():

    read_hist = 10
    negs = 2
    model = CNN('../data/user_history.json', '../data/mat.pkl', read_hist, 20)
    model.create_model()
    model.get_model_summary()
   
    ct = 0
    user_in = []
    article_in = []
    truth = []
    for user in model.users:
        if len(model.user_hist[user]) < 10 or len(model.user_hist[user]) > 15:
            continue
        ct += 1
        temp = []
        user_hist = model.user_hist[user][:-1]
        
        if len(user_hist) > read_hist:
            for art in user_hist[:read_hist]:
                temp.append(model.mat[str(art)])
            for art in user_hist[read_hist:]:
                article_in.append(model.mat[str(art)])
                user_in.append(temp)
                truth.append(1)
                for i in range(negs):
                    article_in.append(model.mat[random.choice(model.articles)])
                    user_in.append(temp)
                    truth.append(0)
 
        else:
            for art in user_hist[:-1]:
                temp.append(model.mat[str(art)])
            for i in range(read_hist-len(temp)):
                temp.append(np.zeros((20,300)))
            article_in.append(model.mat[str(user_hist[-1])])
            user_in.append(temp)
            truth.append(1)
            for i in range(negs):
                article_in.append(model.mat[random.choice(model.articles)])
                user_in.append(temp)
                truth.append(0)
        if ct==250:
            break

    user_in = np.array(user_in)
    article_in = np.array(article_in)
    user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
    article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])

    model.fit_model([user_in, article_in], np.array(truth), 'yo')

def test():
    
    model = CNN('../data/user_history.json', '../data/mat.pkl', 10, 8)
    model.create_model()
    model.get_model_summary()
    model.model.load_weights('../weights/eucl/weights-21-0.32.hdf5')
    
    test_data = np.load('../data/eucl/test_negs.npy')
    truth = np.load('../data/eucl/truth.npy')    
        
    hr = [0]*10
    ndcg = [0]*10

    for i in range(test_data.shape[0]):
        
        data = np.resize(test_data[i], (test_data[i].shape[0], 1) + test_data[i].shape[1:])
        out = model.model.predict([np.array(data)])

        sorted_items = sorted(range(len(out)),key=lambda x:out[x])
        sorted_items.reverse()

        for k in range(10):
            rec = sorted_items[:k+1]
            if 99 in rec:
                hr[k] += 1
            for pos in range(k+1):
                if rec[pos] == 99: 
                    ndcg[k] += 1 / np.log2(1+pos+1)
        
    HR = []
    NDCG = []

    for k in range(10):
        print k, 'hr',  hr[k], 'ndcg', ndcg[k]
        HR.append(float(hr[k]) / float(test_data.shape[0]))
        NDCG.append(float(ndcg[k]) / float(test_data.shape[0]))
        print k, 'HR',  HR[k], 'NDCG', NDCG[k]



def test_run():

    #creating 
    test_model = CNN('../data/user_history.json', '../data/mat.pkl', 10, 8)
    test_model.create_model()
    test_model.get_model_summary()

    no_of_samples = 10
    history = 8

    inputter = []
    for i in xrange(no_of_samples):
        each_sample = []
        final_added = []
        for i in xrange(history):
            each_sample.append(np.random.rand(10, 10))
        final_added.append(each_sample)
        inputter.append(np.array(final_added))
    inputter = np.array(inputter)
    pdb.set_trace()
    print inputter.shape
    output = np.random.randint(2, size=10)

    test_model.test_fit_model([inputter], output)


if __name__ == "__main__":
    train()
