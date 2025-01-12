import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
from seq2seq_model import Encoder, Decoder, Attention  # Assuming models are in seq2seq_model.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EMBED_DIM = 256
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
CLIP = 1  # For gradient clipping

# Prepare data for training
def create_dataloader(src_data, trg_data, batch_size):
    """Create a DataLoader for training"""
    src_tensors = [torch.tensor(sentence, dtype=torch.long) for sentence in src_data]
    trg_tensors = [torch.tensor(sentence, dtype=torch.long) for sentence in trg_data]
    dataset = TensorDataset(
        torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=0),
        torch.nn.utils.rnn.pad_sequence(trg_tensors, batch_first=True, padding_value=0)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(encoder, decoder, train_loader, optimizer, criterion, clip):
    """Training process for one epoch"""
    encoder.train()
    decoder.train()

    epoch_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        # Forward pass through the encoder
        hidden, cell = encoder(src)

        # Initialize decoder input (<sos> token)
        trg_input = trg[:, 0]

        loss = 0
        for t in range(1, trg.size(1)):
            # Decode one token at a time
            output, hidden, cell = decoder(trg_input, hidden, cell, encoder_outputs=hidden)
            loss += criterion(output, trg[:, t])
            
            # Use teacher forcing
            teacher_force = random.random() < 0.5
            top1 = output.argmax(1)
            trg_input = trg[:, t] if teacher_force else top1
        
        # Backpropagation and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() / trg.size(1)

    return epoch_loss / len(train_loader)

def main():
    # Load and preprocess the dataset
    train_data, val_data, test_data = prepare_data('data/eng_fra.parquet', 'data/french_spm_model.model')
    train_loader = create_dataloader(train_data[0], train_data[1], BATCH_SIZE)

    # Initialize encoder, decoder, attention, and optimizer
    src_vocab_size = len(train_data[0])
    trg_vocab_size = len(train_data[1])
    attention = Attention(HID_DIM)
    
    encoder = Encoder(src_vocab_size, EMBED_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(trg_vocab_size, EMBED_DIM, HID_DIM, N_LAYERS, attention, DROPOUT).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_model(encoder, decoder, train_loader, optimizer, criterion, CLIP)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

        # Save model after each epoch
        torch.save(encoder.state_dict(), f'models/encoder_epoch{epoch+1}.pt')
        torch.save(decoder.state_dict(), f'models/decoder_epoch{epoch+1}.pt')

if __name__ == '__main__':
    main()