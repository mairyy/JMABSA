import spacy #spacy v3
nlp = spacy.load("en_core_web_sm")
from typing import DefaultDict
import json

def get_short(path='/Users/admin/Documents/Projects/JMABSA/src/data/twitter2015/test.json', desPath='test.json'):
    data = json.load(open(path, 'r'))
    for row in data:
        words = row['words']

        post_doc = nlp(' '.join(words))
        # post_doc = [nlp(' '.join(word)) for word in words]
        head = []
        # for doc in post_doc:
        #     h = []
        #     for d in doc:
        #         print(d, d.head.i)
        #         h.append(d.head.i)
        #     head.append(h)
        for d in post_doc:
            # print(d, d.head.i)
            head.append(d.head.i)
        # print(head)

        max=len(head)
        tmp = [[0]*max for _ in range(max)]  
        # print(tmp)
        for i in range(max): 
            j=head[i]
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        # print(tmp)
        tmp_dict = DefaultDict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                        leverl_degree[i][q] = 4
                                        #print(word_leverl_degree)
                                        node_set.add(q) 
        # print(leverl_degree)
        row['head'] = head
        row['short'] = leverl_degree
    
    wf = open(desPath, 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

if __name__ == '__main__':
    get_short()
    get_short('/Users/admin/Documents/Projects/JMABSA/src/data/twitter2015/train.json', 'train.json')
    get_short('/Users/admin/Documents/Projects/JMABSA/src/data/twitter2015/dev.json', 'dev.json')