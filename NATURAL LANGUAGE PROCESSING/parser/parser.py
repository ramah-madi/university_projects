from supar import Parser
parser = Parser.load('biaffine-dep-en')
dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=False)

dataset[0]

import torch
print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")






import os
import tempfile
import spacy

path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
 
with open(path, 'w') as f:
  f.write('''# text = But I found the location wonderful and the neighbors very kind.
1\tBut\t_\t_\t_\t_\t_\t_\t_\t_
2\tI\t_\t_\t_\t_\t_\t_\t_\t_
3\tfound\t_\t_\t_\t_\t_\t_\t_\t_
4\tthe\t_\t_\t_\t_\t_\t_\t_\t_
5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_
6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_
7\tand\t_\t_\t_\t_\t_\t_\t_\t_
7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_
8\tthe\t_\t_\t_\t_\t_\t_\t_\t_
9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_
10\tvery\t_\t_\t_\t_\t_\t_\t_\t_
11\tkind\t_\t_\t_\t_\t_\t_\t_\t_
12\t.\t_\t_\t_\t_\t_\t_\t_\t_

''')

dep.predict(path, pred='pred.conllx', verbose=False)[0]

con = Parser.load('crf-con-en')
con.predict(['I', 'saw', 'Sarah', 'with', 'a', 'telescope', '.'], verbose=False)[0].pretty_print()


