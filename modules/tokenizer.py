import jieba
from gensim import corpora
import os
import torch

class Tokenizer():
    def __init__(self, dictionary_path=None,dataset_path="../datasets/vid/"):
        if dictionary_path == None:
            self.dictionary = corpora.Dictionary()
            self.dictionary.token2id['<UNKNOWN>'] = 1
            self.dictionary.token2id['<PAD>'] = 0
            self.dictionary.token2id["<SPAN>"]=2
            self.dictionary.token2id["<SEP>"] = 3


            self.read_dataset(dataset_path)

            self.save_dict(self.dictionary)

        else:
            self.dictionary = corpora.Dictionary.load(dictionary_path)

    def encode(self, texts,debug=0):
        if isinstance(texts,str):
            texts=[texts]
        id_tokens = []
        for text in texts:
            tokenized_text = self.preprocess(text)
            if debug==1:
                print(tokenized_text)
            id_tokens.append(self.dictionary.doc2idx(tokenized_text, unknown_word_index=1))

        return id_tokens

    def preprocess(self, text):
        seg_list = jieba.cut(text)
        # seg_list = [token for token in seg_list if token not in self.stop_words]
        return list(seg_list)

    def save_dict(self, dictionary, path="../models/dictionary.gensim"):
        dictionary.save(path)

    def __len__(self):
        return len(self.dictionary)

    def read_dataset(self, dataset_path):
        dir_names=os.listdir(dataset_path)

        for dir_name in dir_names:
            data_path=os.path.join(dataset_path,dir_name)
            for data in os.listdir(data_path):
                if data.endswith(".txt"):
                    with open(os.path.join(data_path,data),"r",encoding="utf-8") as f:
                        content=f.readline().strip("\n")
                        token = self.preprocess(content)
                        self.dictionary.add_documents([token])

if __name__ == '__main__':
    tokenizer=Tokenizer()
    token=tokenizer.encode("将茶被放到茶托中的股份",debug=1)

    print(token)

    print(tokenizer.dictionary.token2id["<SEP>"])
    print(tokenizer.dictionary.token2id["<UNKNOWN>"])
