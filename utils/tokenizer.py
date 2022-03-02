import json

class WPMTokenizer:
    def __init__(self, vocab_file):

        with open(vocab_file, "r") as f:
            self.encoder = json.load(f)
        
        self.sos_token = "<sos>"
        self.sos_token_id = self.encoder["<sos>"]
        
        self.eos_token = "<eos>"
        self.eos_token_id = self.encoder["<eos>"]
        
        self.unk_token = "<unk>"
        self.unk_token_id = self.encoder["<unk>"]
        
        self.pad_token = "<pad>"
        self.pad_token_id = self.encoder["<pad>"]
        
        self.mask_token = "<mask>"
        self.mask_token_id = self.encoder["<mask>"]

        self.vocabs = [ item[0] for item in sorted(self.encoder.items(), key=lambda x:x[1]) ]
        self.decoder = dict([[item[1], item[0]] for item in self.encoder.items()])

    def tokenize(self, text):
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.
        tokenize function should return `input_ids`, and `attention_mask`
        """
        token_lst = text.split()
        size = len(token_lst)
        input_ids = []
        attention_mask = [False for _ in range(size)]
        
        for token in token_lst:
            token_id = self.convert_token_to_id(token)
            input_ids.append(token_id)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

if __name__ == "__main__":
    tokenizer = WPMTokenizer("./dataset/en2de/vocab/vocab.50K.en")
    print(tokenizer.tokenize("i am not good"))