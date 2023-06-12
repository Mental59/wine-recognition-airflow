import re
from typing import List
import torch
from torch.utils.data import Dataset
from num2words import num2words

__all__ = [
    'CustomDataset'
]

class CustomDataset(Dataset):
    """
    CustomDataset
    returns sentence, tags, mask, custom_features if freq_dict was provided
    """

    num_regex = re.compile(r"^\d+(\.\d+)?$")
    fractional_number_regex = re.compile(r"^\d+/\d+$")

    def __init__(
            self,
            data,
            tag_to_ix,
            word_to_ix,
            case_sensitive=True,
            prepare_dataset=True,
            convert_nums2words=False
    ):
        super(CustomDataset, self).__init__()
        self.case_sensitive = case_sensitive
        self.convert_nums2words = convert_nums2words
        self.pad = 'PAD'
        self.unk = 'UNK'
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        if not case_sensitive:
            self.word_to_ix = {word.lower(): index for word, index in word_to_ix.items()}
            self.pad = self.pad.lower()
            self.unk = self.unk.lower()
        self.raw_data_cached = list(self.compute_raw_data(data))
        self.data = self.prepare_dataset() if prepare_dataset else None

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        mask = tags >= 0
        return sentence, tags, mask
    
    def __len__(self):
        return len(self.data)

    def prepare_dataset(self):
        prepared_data = []
        max_len = self.compute_max_sentence_len(self.raw_data_cached)
        for sentence, tags in self.raw_data_cached:
            sentence = self.sentence_to_indices(sentence)
            tags = self.tags_to_indices(tags)

            sentence.extend([self.word_to_ix[self.pad]] * (max_len - len(sentence)))
            tags.extend([-1] * (max_len - len(tags)))

            prepared_data.append(
                (torch.LongTensor(sentence), torch.LongTensor(tags))
            )
        return prepared_data

    def sentence_to_indices(self, sentence):
        return [
            self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.unk] for word in sentence
        ]

    def tags_to_indices(self, tags):
        return [self.tag_to_ix[tag] for tag in tags]

    @staticmethod
    def number2words(number) -> List[str]:
        return (
            num2words(number)
            .replace('-', ' ')
            .replace(',', '')
            .split()
        )

    def compute_raw_data(self, data):
        for sentence, tags in data:
            if not self.case_sensitive:
                sentence = [word.lower() for word in sentence]

            if self.convert_nums2words:
                new_sentence = []
                new_tags = []

                for index, (word, tag) in enumerate(zip(sentence, tags)):
                    if CustomDataset.num_regex.match(word):
                        words = CustomDataset.number2words(word)
                        new_sentence.extend(words)
                        new_tags.extend([tag] * len(words))
                    elif CustomDataset.fractional_number_regex.match(word):
                        first_price, second_price = word.split('/')
                        first_price_words = CustomDataset.number2words(first_price)
                        second_price_words = CustomDataset.number2words(second_price)

                        new_sentence.extend(first_price_words)
                        new_tags.extend([tag] * len(first_price_words))

                        new_sentence.append('/')
                        new_tags.append('Other')

                        new_sentence.extend(second_price_words)
                        new_tags.extend([tag] * len(second_price_words))
                    else:
                        new_sentence.append(word)
                        new_tags.append(tag)

                sentence = new_sentence
                tags = new_tags

            yield sentence, tags

    def raw_data(self):
        return self.raw_data_cached

    @staticmethod
    def compute_max_sentence_len(data):
        return len(max(data, key=lambda x: len(x[0]))[0])

    def count(self, word: str):
        if not self.case_sensitive:
            word = word.lower()
        if word not in self.word_to_ix:
            raise ValueError(f'Tag: {word} is not in word_to_ix dictionary keys')
        word_ix = self.word_to_ix[word]
        count_word = 0
        for sentence, _ in self.data:
            count_word += torch.sum(sentence == word_ix).item()
        return count_word
