import torch

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
print(data.shape, data.dtype)
print(data[:1000])


n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(98772)
batch_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    xi =  torch.randint(len(data)- block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in xi])
    y = torch.stack([data[i+1:i+block_size] for i in xi])
    return x , y
xb, yb = get_batch('train')
#print('input')
#print(xb)
#print('output')
#print(yb)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        traget = yb[b,t]
        print(f'when the input is {context.tolist()} the  target {target}')
