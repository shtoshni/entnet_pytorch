import sys
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from memory import DynamicMemory

class EntNet(nn.Module):
    def __init__(self, vocab_size, memory_slots=20, emb_size=100, bow_encoding=False, max_sentence_length=50, max_query_length=50):
        super(EntNet, self).__init__()
        self.bow_encoding = bow_encoding

        self.memory_slots = memory_slots
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        # Initialize embedding
        emb_init_weight = nn.init.normal_(torch.empty(vocab_size, emb_size), mean=0, std=0.1)
        emb_init_weight[0].zero_()  # 0 val for padding symbol
        self.embedding = nn.Embedding(vocab_size, emb_size, _weight=emb_init_weight)

        # Make padding synbol gradient zero
        def emb_hook(grad):
            grad[0].zero_()
            return grad
        self.embedding.weight.register_hook(lambda grad: emb_hook(grad))

        # Initialize Sentence Encoder Parameters with all ones (BoW)
        init_story_weight = torch.ones(max_sentence_length, emb_size)
        init_query_weight = torch.ones(max_query_length, emb_size)
        if self.bow_encoding:
            self.mult_mask = nn.Embedding.from_pretrained(init_story_weight, freeze=True)
            self.mult_mask_query = nn.Embedding.from_pretrained(init_query_weight, freeze=True)
        else:
            self.mult_mask = nn.Embedding(max_sentence_length, emb_size, _weight=init_story_weight)
            self.mult_mask_query = nn.Embedding(max_query_length, emb_size, _weight=init_query_weight)

        # Initialize the PRelu activation
        self.activation = nn.PReLU(num_parameters=emb_size, init=1.0)

        self.memory_net = DynamicMemory(hidden_size=emb_size, memory_slots=memory_slots, activation=self.activation)

        # Output module parameters
        # Initialize R
        R_init_weight = nn.init.normal_(torch.empty(emb_size, vocab_size), mean=0, std=0.1)
        # Zeroout the 0th column representing output embedding for pad
        R_init_weight[:, 0].zero_()
        self.R = Parameter(R_init_weight)

        # Make gradient corresponding to padding symbol 0
        def R_hook(grad):
            grad[:, 0].zero_()
            return grad
        self.R.register_hook(lambda grad: R_hook(grad))

        self.H = Parameter(nn.init.normal_(torch.empty(emb_size, emb_size), mean=0, std=0.1))


    def encode_sent(self, batch_sent, query=False):
        """
        Encode a batch of sentences.
        batch_sent: B x L
        """
        # Embed Sentence
        sent_tensor = self.embedding(batch_sent)  # B x L x E
        # Get non-zero indices (Will only select indices >= 1)
        mask_tensor = torch.unsqueeze(torch.ge(batch_sent, 1), dim=2).cuda()  # B x L x 1
        # Cast the mask_tensor
        mask_tensor = mask_tensor.type(dtype=torch.cuda.FloatTensor)
        # Mask out padded symbols
        sent_tensor = sent_tensor * mask_tensor # B x L x E

        # Get embedding for multiplicative masks
        _, sentence_length = list(batch_sent.size())
        idx_tensor = torch.arange(sentence_length).cuda()
        if query:
            mult_mask_tensor = self.mult_mask_query(idx_tensor)
        else:
            mult_mask_tensor = self.mult_mask(idx_tensor) # L x E

        batch_sent_emb = sent_tensor * mult_mask_tensor # B x L x E
        batch_sent_emb = torch.sum(batch_sent_emb, 1) # B x E

        return batch_sent_emb


    def encode_story(self, batch_story):
        """Encode a batch of stories.
        batch_story: B x T x L (T is # of sentences and L is max length of each sentence)

        Return:-
        enc_stories: list of B x H tensors with length T.
        mask_stories: list of length T with tensors of size (B,)
        """
        batch_story = torch.transpose(batch_story, 0, 1)  # T x B x L

        # Split stories along sentences
        seq_batch_sent = torch.unbind(batch_story, dim=0)  # Tuple of B x L tensors
        enc_stories = []
        mask_stories = []
        for batch_sent in seq_batch_sent:
            enc_stories.append(self.encode_sent(batch_sent))
            # Check if all the symbols are pad or not
            mask_at_t = torch.ge(torch.sum(batch_sent, dim=1), 1.0).type(dtype=torch.cuda.FloatTensor)
            mask_stories.append(mask_at_t)
        return enc_stories, mask_stories


    def read_story_and_answer_question(self, batch_story, batch_question):
        """Read stories via memory network and answer related questions.
        batch_story: B x T x L (T is # of sentences and L is sentence length)
        batch_question: B x L

        Return:- answer
        """
        enc_stories, mask_stories = self.encode_story(batch_story)
        enc_query = self.encode_sent(batch_question, query=True)  # B x H

        memorized_stories = self.memory_net.read_story(enc_stories, mask_stories)  # M x B x H

        # Get softmax scores for memories
        memory_scores = torch.sum(memorized_stories * enc_query, dim=2)  # M x B
        softmax_scores = nn.functional.softmax(memory_scores, dim=0) # M x B

        # Get story representation corresponding to query
        softmax_scores = torch.unsqueeze(softmax_scores, 2)  # M x B x 1
        weighted_memories = softmax_scores * memorized_stories  # M x B x H
        story_repr = torch.sum(weighted_memories, dim=0)  # B x H

        # Output scores
        activation_input = enc_query + torch.mm(story_repr, self.H)  # B x H
        pred = torch.mm(self.activation(activation_input), self.R)  # B x V

        return pred

    def get_loss(self, preds, target):
        """Return the loss function given the prediction and correct labels.
        preds: B x V
        target: B (integer valued)
        """
        loss = nn.functional.cross_entropy(input=preds, target=target)
        return loss
