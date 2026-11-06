class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.evt_token_id = tokenizer.convert_tokens_to_ids('[EVT]')

    def encode(self, text, length, positions=()):
        segments = [text[i:i + length] for i in range(0, len(text), length)]
        input_ids = [[self.bos_token_id]] + self.tokenizer(segments)['input_ids']
        event_ids = [[self.evt_token_id] * positions.count(i) for i in range(len(input_ids))]
        token_ids = []
        for x, y in zip(input_ids, event_ids):
            token_ids += x + y
        return token_ids
