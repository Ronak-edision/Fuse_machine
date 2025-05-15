# src/backend/model.py
import torch
import torch.nn as nn
import pickle
import math
from . import config
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=33):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head, batch_first=True)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_mask(self, tgt_seq_len, device):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
        return tgt_mask

    def forward(self, encoded_image, decoder_inp):
        device = decoder_inp.device

        decoder_inp_embed = self.embedding(decoder_inp) * math.sqrt(self.embedding_size)
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)

        tgt_mask = self.generate_mask(decoder_inp.size(1), device)
        tgt_key_padding_mask = (decoder_inp == 0)  # Assuming 0 is the padding index

        decoder_output = self.TransformerDecoder(
            tgt=decoder_inp_embed,
            memory=encoded_image,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        final_output = self.last_linear_layer(decoder_output)
        return final_output

def load_model_and_vocab():
    config.logger.info(f"Loading vocabulary from: {config.VOCAB_PATH}")
    with open(config.VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
        word_to_index = vocab_data["vocab_to_index"]
        index_to_word = vocab_data["index_to_vocab"]
        vocab_size = len(word_to_index)
    config.logger.info(f"Vocabulary loaded. Size: {vocab_size}")

    model = ImageCaptionModel(n_head=16, n_decoder_layer=4, vocab_size=vocab_size, embedding_size=512)
    config.logger.info(f"Loading model state_dict from: {config.MODEL_PATH}")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    config.logger.info("Model loaded and set to evaluation mode.")
    return model, word_to_index, index_to_word, vocab_size

def generate_caption(model, word_to_index, index_to_word, vocab_size, image_name, max_seq_len=33):
    device = torch.device("cpu")

    config.logger.info(f"Generating caption for image: {image_name}")
    config.logger.debug(f"Loading embeddings from: {config.ENCODED_IMAGE_VAL_PATH}")
    with open(config.ENCODED_IMAGE_VAL_PATH, "rb") as f:
        valid_img_embed_all = pickle.load(f)

    if image_name not in valid_img_embed_all:
        err_msg = f"Image {image_name} not found in embeddings file: {config.ENCODED_IMAGE_VAL_PATH}"
        config.logger.error(err_msg)
        raise ValueError(err_msg)

    img_embed_raw = valid_img_embed_all[image_name]

    img_embed = img_embed_raw.to(device)
    if img_embed.ndim == 3:
        img_embed = img_embed.unsqueeze(0)

    batch_size, channels, height, width = img_embed.shape
    img_embed = img_embed.permute(0, 2, 3, 1)
    img_embed = img_embed.reshape(batch_size, height * width, channels)

    start_token_idx = word_to_index['<start>']
    eos_token_idx = word_to_index['<eos>']
    pad_token_idx = word_to_index.get('<pad>', 0)

    input_seq = torch.tensor([[start_token_idx]], device=device)
    predicted_indices = []
    model.eval()

    with torch.no_grad():
        for _ in range(max_seq_len - 1):
            output_logits = model(img_embed, input_seq)
            last_token_logits = output_logits[:, -1, :]

            last_token_logits[:, pad_token_idx] = -float('inf')

            _, next_word_idx_tensor = torch.topk(last_token_logits, 1, dim=-1)
            next_word_idx = next_word_idx_tensor.item()

            if next_word_idx == eos_token_idx:
                break

            predicted_indices.append(next_word_idx)
            input_seq = torch.cat([input_seq, next_word_idx_tensor], dim=1)

    predicted_sentence = [index_to_word.get(idx, "<unk>") for idx in predicted_indices]
    final_caption = " ".join(predicted_sentence)
    config.logger.info(f"Generated caption for {image_name}: {final_caption}")
    return final_caption