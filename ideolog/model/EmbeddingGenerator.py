from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch


class EmbeddingGenerator:
    def __init__(self):
        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.config.output_hidden_states = True

        self.tok = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base", config=self.config)

    def get_embeddings(self, prompt, id=None, willSave=True):
        input_tensor = torch.tensor([self.tok.encode(prompt, max_length=512, padding="max_length", truncation=True)])
        output = self.model(input_tensor)  # returns a tuple(sequence_output, pooled_output, hidden_states)
        hidden_states = output[-1]

        embedding_output = hidden_states[0]

        if willSave:
            if id is None:
                raise ValueError('asked to save tensor but no id given')
            torch.save(embedding_output, './data/tensors/' + id + '.pt')

        return embedding_output
