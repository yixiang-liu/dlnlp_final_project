import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq_model import Encoder, Decoder, Attention
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 翻译句子
def translate_sentence(encoder, decoder, sentence, src_vocab, trg_vocab, max_len=50):
    encoder.eval()
    decoder.eval()

    # 将句子转换为索引
    src_indexes = [src_vocab.stoi[token] for token in sentence]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = encoder(src_tensor)

    trg_indexes = [trg_vocab.stoi['<sos>']]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)
        with torch.no_grad():
            output, hidden, cell = decoder(trg_tensor, hidden, cell, encoder_outputs=hidden)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.stoi['<eos>']:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:-1]  # 移除 <sos> 和 <eos>

# 计算 BLEU 分数
def calculate_bleu(encoder, decoder, test_data, src_vocab, trg_vocab):
    total_bleu_score = 0
    for src_sentence, trg_sentence in zip(test_data[0], test_data[1]):
        predicted_trg = translate_sentence(encoder, decoder, src_sentence, src_vocab, trg_vocab)
        reference = [trg_sentence]
        bleu_score = sentence_bleu(reference, predicted_trg)
        total_bleu_score += bleu_score

    return total_bleu_score / len(test_data[0])

# 计算困惑度
def calculate_perplexity(loss):
    return math.exp(loss)

def main():
    # 参数和模型初始化
    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)
    EMBED_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5

    attention = Attention(HID_DIM)
    encoder = Encoder(input_dim, EMBED_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(output_dim, EMBED_DIM, HID_DIM, N_LAYERS, attention, DROPOUT).to(device)

    encoder.load_state_dict(torch.load('models/encoder_epoch10.pt'))
    decoder.load_state_dict(torch.load('models/decoder_epoch10.pt'))

    # 加载测试数据和词汇表
    _, _, test_data = prepare_data('data/train-00000-of-00001.parquet', 'data/french_spm_model.model')
    src_vocab = torch.load('eng_vocab.pt')  # 英语词汇表
    trg_vocab = torch.load('fra_vocab.pt')  # 法语词汇表

    # 计算 BLEU 分数
    bleu_score = calculate_bleu(encoder, decoder, test_data, src_vocab, trg_vocab)
    print(f'BLEU Score: {bleu_score:.4f}')

    # 计算困惑度
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi['<pad>'])
    test_loss = 0
    test_loader = create_dataloader(test_data[0], test_data[1], batch_size=32)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            hidden, cell = encoder(src)
            trg_input = trg[:, 0]
            loss = 0
            for t in range(1, trg.size(1)):
                output, hidden, cell = decoder(trg_input, hidden, cell, encoder_outputs=hidden)
                loss += criterion(output, trg[:, t])
                trg_input = trg[:, t]

            test_loss += loss.item() / trg.size(1)

    test_loss /= len(test_loader)
    perplexity = calculate_perplexity(test_loss)
    print(f'Test Loss: {test_loss:.4f}, Perplexity: {perplexity:.4f}')

if __name__ == '__main__':
    main()