import logging
import json
import pickle
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, RandomSampler, BatchSampler
from utils import data2lmdb
from tokenizer import ZhTokenizer
logger = logging.getLogger(__name__)


######################################################################
# The full process for preparing the data is:
#
# -  Read text/question/answer from json
# -  tokenize text using jieba(chinese segmentation) into single chinese char or english word
# -  cut tokenized text by max_len
# -  Make tensor lists from t/q/a pairs
#

def combine_text_answer(text,answer,tokenizer,max_text_len):
    pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer._convert_token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer._convert_token_to_id(tokenizer.eos_token)
    posOfAnswer = text.find(answer)
    if posOfAnswer == -1:
        return None
    # text
    spans = [text[:posOfAnswer],answer, text[posOfAnswer+len(answer):]]
    tokenized_text = []
    token_type_ids = []
    for index,span in enumerate(spans):
        if span:
            tokens = tokenizer.encode(span)
            tokenized_text.extend(tokens)
            token_type_ids.extend([int(index==1)]*len(tokens))
            if index>=1 and len(tokenized_text)>max_text_len-2:
                break
    tokenized_text = tokenized_text[-max_text_len+2:] # truncate from right to left
    tokenized_text = [bos_token_id]+tokenized_text+[eos_token_id]
    tokenized_text_length=len(tokenized_text)
    tokenized_text = tokenized_text + ([pad_token_id] *  (max_text_len - tokenized_text_length) ) # Pad up to the sequence length.  

    assert len(tokenized_text) == max_text_len  
    # token type
    token_type_ids = token_type_ids[-max_text_len+2:] # truncate from right to left
    token_type_ids = [0]+token_type_ids+[0]
    token_type_ids = token_type_ids + ([0] *  (max_text_len - tokenized_text_length) ) # Pad up to the sequence length.  
    assert len(token_type_ids) == max_text_len
    return tokenized_text,token_type_ids

class MedQaDataset(Dataset):
    def __init__(self,tokenizer, file_path='data/round1_train_0907.json', max_text_len=512, max_question_len=128,prefix=''):
        self.pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
        self.bos_token_id = tokenizer._convert_token_to_id(tokenizer.bos_token)
        self.eos_token_id = tokenizer._convert_token_to_id(tokenizer.eos_token)
        self.examples = []
        self.tokenizer = tokenizer
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'{prefix}_{osp.splitext(filename)[0]}.pkl')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            dropped = self.read_corpus(file_path, tokenizer, max_text_len, max_question_len)
            logger.info("The number of dropped sentences is %d", dropped)
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # pdb.set_trace()
        # Convert to Tensors and build dataset
        assert np.max(self.examples[item][0])<self.tokenizer.vocab_size()
        assert np.max(self.examples[item][1])<self.tokenizer.vocab_size()
        assert np.max(self.examples[item][2])<self.tokenizer.vocab_size()
        tokenized_text= torch.tensor(self.examples[item][0], dtype=torch.long)
        tokenized_token_type = torch.tensor(self.examples[item][1], dtype=torch.long)
        tokenized_question = torch.tensor(self.examples[item][2], dtype=torch.long)
        return (tokenized_text, tokenized_token_type,tokenized_question)


    def read_corpus(self, fname, tokenizer , max_text_len, max_question_len):
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_data=[]
        dropped=0
        for line in data:
            text=line['text'].strip()
            for annotation in line['annotations']:
                question=annotation['Q'].strip()
                answer=annotation['A'].strip()
                # process to token ids
                posOfAnswer = text.find(answer)
                if posOfAnswer == -1:
                    dropped += 1
                    continue
                # text
                spans = [text[:posOfAnswer],answer, text[posOfAnswer+len(answer):]]
                tokenized_text = []
                token_type_ids = []
                for index,span in enumerate(spans):
                    if span:
                        tokens = tokenizer.encode(span)
                        tokenized_text.extend(tokens)
                        token_type_ids.extend([int(index==1)]*len(tokens))
                        if index>=1 and len(tokenized_text)>max_text_len-2:
                            break
                tokenized_text = tokenized_text[-max_text_len+2:] # truncate from right to left
                tokenized_text = [self.bos_token_id]+tokenized_text+[self.eos_token_id]
                tokenized_text_length=len(tokenized_text)
                tokenized_text = tokenized_text + ([self.pad_token_id] *  (max_text_len - tokenized_text_length) ) # Pad up to the sequence length.  
                
                assert len(tokenized_text) == max_text_len
                # token type
                token_type_ids = token_type_ids[-max_text_len+2:] # truncate from right to left
                token_type_ids = [0]+token_type_ids+[0]
                token_type_ids = token_type_ids + ([0] *  (max_text_len - tokenized_text_length) ) # Pad up to the sequence length.  
                assert len(token_type_ids) == max_text_len
                # question
                tokenized_question = tokenizer.encode(question)
                tokenized_question = tokenized_question[-max_question_len+2:] # truncate from right to left
                tokenized_question = [self.bos_token_id]+tokenized_question+[self.eos_token_id]
                tokenized_question_length=len(tokenized_question)
                tokenized_question = tokenized_question + ([self.pad_token_id] *  (max_question_len - tokenized_question_length) ) # Pad up to the sequence length.      
                assert len(tokenized_question) == max_question_len
                self.examples.append([tokenized_text,token_type_ids,tokenized_question])
        return dropped

def train_test_split():
    import random
    with open('data/round1_train_0907.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        num_samples = len(data)
        indexTrain = int(num_samples*0.9)
        random.shuffle(data)
        train = data[:indexTrain]
        valid = data[indexTrain:]
    with open('data/round1_train_0920.json', 'w', encoding='utf-8') as f:
        json.dump(train,f,indent=2)
    with open('data/round1_valid_0920.json', 'w', encoding='utf-8') as f:
        json.dump(valid,f,indent=2)

if __name__ == "__main__":
    # train_test_split()
    tokenizer = ZhTokenizer('vocab.txt',split_by_literals=True)
    dataset = MedQaDataset(tokenizer,prefix='zhtokenizer',file_path='data/round1_train_0920.json')
    print('train examples#:',len(dataset))
    print(dataset[0])
    
    dataset = MedQaDataset(tokenizer,prefix='zhtokenizer',file_path='data/round1_valid_0920.json')
    print('valid examples#:',len(dataset))
    print(dataset[0])
    
    dataset = MedQaDataset(tokenizer,prefix='zhtokenizer',file_path='data/round1_test_0907.json')
    print('test examples#:',len(dataset))
    print(dataset[0])

    combined_result = combine_text_answer('橄榄，又名青果、白榄，为橄榄科植物橄榄的果实','橄榄科植物',tokenizer,64)
    print('random input:',combined_result)
    tokenizer.save_vocab('vocab.txt')