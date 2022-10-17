import pickle
import sqlite3
import numpy as np
import os

vectorizer = pickle.load(open('tranform.pkl','rb'))


def update_model(db_path, model, batch_size=10000):
    label = {'negative': 0, 'positive': 1}
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = np.array([label[i] for i in data[:,1]])

        classes = np.array(['negative', 'positive'])
        X_train = vectorizer.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model

cur_dir = os.path.dirname(__file__)

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))


db = os.path.join(cur_dir, 'reviews.sqlite')

clf = update_model(db_path=db, model=clf, batch_size=10000)