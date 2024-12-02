import json
import pickle
# full = json.load(open('src/data/twitter2015/train_full.json', 'r'))
# few = json.load(open('src/data/twitter2015/train.json', 'r'))

# image_ids = [f['image_id'] for f in few]
# new = []

# for row in full:
#     if row['image_id'] in image_ids:
#         new.append(row)

# wf = open('train.json', 'w')
# wf.write(json.dumps(new, indent=4))
# wf.close()

with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[0])