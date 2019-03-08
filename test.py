# from bert_serving.client import BertClient
# bc = BertClient()
# a = bc.encode(['First do it', 'then do it right', 'then do it better'])
# print(a[0].shape)
# print(len(a[0]))
# print(a)
# print(type(a))


import torch
import torch.nn.functional as F
a = torch.zeros(128,10,1)
b = F.softmax(a.view(-1,10),dim=1)


print(b.shape)
c = b.unsqueeze(1)
print(c.shape)