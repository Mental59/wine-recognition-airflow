import os
import json
import torch
from torch.utils.data import Dataset

__all__ = [
    'CustomDataset'
]

class CustomDataset(Dataset):
    """
    CustomDataset
    returns sentence, tags, mask, custom_features if freq_dict was provided
    """

    def __init__(
            self,
            data_path,
            data_length,
            max_sent_length,
            tag_to_ix,
            word_to_ix,
            case_sensitive=True
    ):
        super(CustomDataset, self).__init__()
        self.case_sensitive = case_sensitive
        self.pad = 'PAD'
        self.unk = 'UNK'
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

        if not case_sensitive:
            self.word_to_ix = {word.lower(): index for word, index in word_to_ix.items()}
            self.pad = self.pad.lower()
            self.unk = self.unk.lower()
            
        self.data_length = data_length
        self.data_path = data_path
        self.max_sent_length = max_sent_length

    def __getitem__(self, index):
        sentence, tags = self.load_data(index)
        sentence, tags = self.prepare_data(sentence, tags)
        mask = tags >= 0
        f = torch.empty(0)
        return sentence, tags, mask, f
    
    def __len__(self):
        return self.data_length
    
    def load_data(self, index):
        with open(os.path.join(self.data_path, f'{index}.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['words'], data['tags']
    
    def prepare_data(self, sentence, tags):
        sentence = self.sentence_to_indices(sentence)
        tags = self.tags_to_indices(tags)

        sentence.extend([self.word_to_ix[self.pad]] * (self.max_sent_length - len(sentence)))
        tags.extend([-1] * (self.max_sent_length - len(tags)))

        return torch.LongTensor(sentence), torch.LongTensor(tags)

    def sentence_to_indices(self, sentence):
        return [
            self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.unk] for word in sentence
        ]

    def tags_to_indices(self, tags):
        return [self.tag_to_ix[tag] for tag in tags]

    def raw_data(self):
        return (self.load_data(i) for i in range(self.data_length))

    # def count(self, word: str):
    #     if not self.case_sensitive:
    #         word = word.lower()
    #     if word not in self.word_to_ix:
    #         raise ValueError(f'Tag: {word} is not in word_to_ix dictionary keys')
    #     word_ix = self.word_to_ix[word]
    #     count_word = 0
    #     for sentence, _ in self.data:
    #         count_word += torch.sum(sentence == word_ix).item()
    #     return count_word
