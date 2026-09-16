import torch
from torch import nn

from model.features_extraction import ImageEmbedding, QuesEmbedding, AnsEmbedding
from model.sans import StackAttention
from model.decoder_model import Decoder
from configs.config import Config


class VQAModel(nn.Module):
    """
    Visual Question Answering Model utilizing Optimal Transport concepts (via SAN).

    Architecture:
        +-------------------+   +-----------------------+
        | Image (DeiT)      |   | Question (PhoBERT)    |
        +--------+----------+   +-----------+-----------+
                 |                          |
                 |       +------------+     |
                 +-----> |    SAN     | <---+
                         | (Attn &    |
                         |  Fusion)   |
                         +-----+------+
                               |
                               v
                      [ Context Memory ]
                               |
                               v
                       +---------------+   +-----------------------+
                       |               |   | Answer Prefix Target  |
                       |   Decoder     | < + (PhoBERT Embeddings)  |
                       |               |   +-----------------------+
                       +-------+-------+
                               |
                               v
                      [ Next Token Logits ]
    """
    def __init__(self, vocab_size=None, output_size=768, d_model=768,
                 num_heads=4, ffn_hidden=2048, drop_prob=0.1, num_layers=4,
                 num_att_layers=2, mode='train', text_model=Config.textmodel_dir,
                 image_model=Config.image_model):
        super().__init__()
        if output_size != d_model or num_att_layers < 1:
            raise ValueError('output_size must equal d_model and at least one attention layer is needed')
        self.mode = mode
        self.model_config = dict(output_size=output_size, d_model=d_model, num_heads=num_heads,
                                 ffn_hidden=ffn_hidden, drop_prob=drop_prob, num_layers=num_layers,
                                 num_att_layers=num_att_layers)
        self.image_model = ImageEmbedding(image_model)
        self.ques_model = QuesEmbedding(output_size=output_size, model_name=text_model)
        self.ans_model = AnsEmbedding(model_name=text_model)
        self.tokenizer = self.ans_model.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                             else self.tokenizer.cls_token_id)
        self.eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None
                             else self.tokenizer.sep_token_id)
        if any(value is None for value in (self.pad_token_id, self.bos_token_id, self.eos_token_id)):
            raise ValueError('Tokenizer needs PAD, BOS/CLS and EOS/SEP tokens')
        image_dim = self.image_model.model.config.hidden_size
        answer_dim = self.ans_model.phobert_embed.word_embeddings.embedding_dim
        self.image_projection = nn.Identity() if image_dim == d_model else nn.Linear(image_dim, d_model)
        self.answer_projection = nn.Identity() if answer_dim == d_model else nn.Linear(answer_dim, d_model)
        self.san_model = nn.ModuleList([StackAttention(d=d_model, k=512) for _ in range(num_att_layers)])
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        actual_vocab = self.ans_model.phobert_embed.word_embeddings.num_embeddings
        if vocab_size is not None and vocab_size != actual_vocab:
            raise ValueError('vocab_size must match the text model embedding vocabulary')
        self.mlp = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(d_model, d_model),
                                 nn.GELU(), nn.Linear(d_model, actual_vocab))

    def encode(self, images, questions, anno_ids=None):
        image_embeddings, _ = self.image_model(images, image_ids=anno_ids)
        # DeiT already returns [batch, image_tokens, hidden]; reshaping corrupts it.
        image_embeddings = self.image_projection(image_embeddings)
        context = self.ques_model(questions)
        for layer in self.san_model:
            context = layer(image_embeddings, context.unsqueeze(1))
        return context.unsqueeze(1)

    def decode(self, input_ids, memory, causal=True):
        target = self.answer_projection(self.ans_model.embed_ids(input_ids))
        length = input_ids.size(1)
        blocked = input_ids.eq(self.pad_token_id)[:, None, None, :].expand(-1, 1, length, -1)
        if causal:
            blocked = blocked | torch.triu(torch.ones(length, length, dtype=torch.bool,
                                                     device=input_ids.device), diagonal=1)
        # Decoder expects (encoder memory, answer target, self-attention mask).
        return self.mlp(self.decoder(memory, target, blocked))

    def forward(self, images, questions, answers=None, anno_ids=None, mask=True,
                mode='train', max_len=Config.MAX_LEN_ANS):
        if mode not in {'train', 'eval', 'test', 'infer', 'generate'}:
            raise ValueError(f'Unknown mode: {mode}')
        if mode in {'test', 'infer', 'generate'} or answers is None:
            return self.generate(images, questions, anno_ids, max_len)
        memory = self.encode(images, questions, anno_ids)
        ids = self.ans_model.tokenize(answers, max_len)
        return self.decode(ids[:, :-1], memory, causal=mask), ids[:, 1:]

    @torch.no_grad()
    def generate(self, images, questions, anno_ids=None, max_len=Config.MAX_LEN_ANS):
        if not 2 <= max_len <= Config.MAX_LEN_ANS:
            raise ValueError(f'max_len must be between 2 and {Config.MAX_LEN_ANS}')
        was_training = self.training
        self.eval()
        try:
            memory = self.encode(images, questions, anno_ids)
            ids = torch.full((images.size(0), 1), self.bos_token_id, dtype=torch.long, device=memory.device)
            finished = torch.zeros(images.size(0), dtype=torch.bool, device=memory.device)
            for _ in range(max_len - 1):
                logits = self.decode(ids, memory)[:, -1, :]
                logits[:, [self.pad_token_id, self.bos_token_id]] = float('-inf')
                next_ids = logits.argmax(-1).masked_fill(finished, self.pad_token_id)
                ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
                finished |= next_ids.eq(self.eos_token_id)
                if finished.all():
                    break
            return ids[:, 1:]
        finally:
            self.train(was_training)

    def answers_from_ids(self, ids):
        rows = []
        for row in ids.detach().cpu().tolist():
            if self.eos_token_id in row:
                row = row[:row.index(self.eos_token_id)]
            rows.append(row)
        return self.tokenizer.batch_decode(rows, skip_special_tokens=True)
