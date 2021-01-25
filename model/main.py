import dill
import os
import pickle
import tensorflow as tf
import re

tf.logging.set_verbosity(tf.logging.ERROR)

with open('/artifacts/ScoringService.dl', 'rb') as fout:
    s = dill.load(fout)

s.get_model('/artifacts')

print(f"Health Check: {s.health_check()}")

print("Prediction:")
queries = ['SELECT * FROM customer', 'SELECT * FROM customer']
print(f"Queries: {queries}")
result = s.predict(queries)
print(f"Result: {result}")
