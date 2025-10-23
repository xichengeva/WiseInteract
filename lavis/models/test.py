from transformers import BertTokenizerFast, BertModel
checkpoint = '/home/datahouse1/niubuying/CPI/LAVIS/models/bert-base-smiles'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
model = BertModel.from_pretrained(checkpoint)

example = 'O=C([C@@H](c1ccc(cc1)O)N)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C'
tokens = tokenizer(example, return_tensors='pt')
print(tokens)
predictions = model(**tokens)
print(predictions)