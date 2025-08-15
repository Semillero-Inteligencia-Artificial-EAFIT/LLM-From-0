import torch
import torch.nn as nn
from torch.nn import functional as F 

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

text = read_txt_file("example.txt")
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

#encode=encode("hello")
#decode=decode(encode)
#print(decode)

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])


n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

block_size = 64

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(42)
batch_size = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    xi =  torch.randint(len(data)- block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in xi])
    y = torch.stack([data[i+1:i+block_size+1] for i in xi])
    return x , y
xb, yb = get_batch('train')

#print('input')
#print(xb)
#print('output')
#print(yb)

'''
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t].item()
        print(f'when the input is {context.tolist()} the  target {target}')
        print()
        print(f'when the input is {decode(context.tolist())} the  target {itos[target]}')
        print()
'''


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get prediction
            logits, _ = self(idx)  # logits = (B,T,C)
            # Focus on last token
            logits = logits[:, -1, :]  # (B,C)
            # Apply softmax
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx

# Initialize model
model = BigramLanguageModel(vocab_size)

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 64
max_iters = 20000  # Reduced for demonstration

for steps in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print(f"Step {steps}: Loss {loss.item():.4f}")

# Test generation with user input
def generate_from_input():
    while True:
        user_input = input("\nEnter starting text (or 'quit' to exit): ").lower()
        if user_input.lower() == 'quit':
            break
            
        # Encode input and convert to tensor
        start_tokens = encode(user_input)
        context = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)
        
        # Generate tokens
        generated = model.generate(context, max_new_tokens=100)
        generated_text = decode(generated[0].tolist())
        
        # Highlight input vs generated text
        print("\n" + "-"*50)
        print(f"Input:    '{user_input}'")
        print(f"Generated: '{generated_text}'")
        print("-"*50 + "\n")

# Start interactive session
print("\n" + "="*60)
print("Text Generation Started! (type 'quit' to exit)")
print("="*60)
generate_from_input()