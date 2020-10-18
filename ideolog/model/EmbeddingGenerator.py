from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import os

class EmbeddingGenerator:
    ''' A class which generates embeddings for text and saves them to data/tensors/ '''

    def __init__(self):
        self.config = RobertaConfig.from_pretrained("roberta-base")
        self.config.output_hidden_states = True

        self.tok = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base", config=self.config)

    def get_embeddings(self, prompt, id=None, willSave=True):
        '''
        Generates embeddings for a prompt based on roberta.

        :param prompt: The body of text to get embeddings from.
        :param id: The id of the bill, used to determine a filename for the tensor to be saved in. Only needed if
        willSave is True.
        :param willSave: Whether or not to save the embeddings generated for this prompt. Use True if you are creating
        tensors for bills and use false if it is a tensor for a prompt
        :return:
        '''
        input_tensor = torch.tensor([self.tok.encode(prompt, max_length=512, padding="max_length", truncation=True)])
        output = self.model(input_tensor)  # returns a tuple(sequence_output, pooled_output, hidden_states)
        hidden_states = output[-1]

        embedding_output = hidden_states[0]
        if not os.path.isdir("../../data/tensors"):
            os.mkdir("../../data/tensors")

        if willSave:
            if id is None:
                raise ValueError('asked to save tensor but no id given')
            torch.save(embedding_output, '../../data/tensors/' + id + '.pt')

        return embedding_output
