import sqlite3
import os

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS review_db')
c.execute('CREATE TABLE review_db''(review TEXT, sentiment INTEGER, date TEXT)')