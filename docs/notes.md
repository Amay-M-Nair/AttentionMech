## Week 1 : Basic Attention

### Setup 
- Keys: represent the words we have
- Query: represents what we're looking for
- Both are vectors (2D for simplicity)

### Attention Formula - step 1
- scores = query @ keys.T
- Higher score = more relevant
- Shape: (num_keys,)

### Attention Formula - step 2
- attention = F.softmax(score, dim=0)
- Softmax converts scores to probabilities. dim = 0 means softmax in 'rows'
- All weights sum to 1.0
- Can intepret as "percentage of focus"

### Attention Formula - step 3
- output = attention @ keys

## Batch Processing
- queries = torch.tensor([[...],[...]]) #Shape: (num_queries,dim)
- scores = queries @ keys.T             #Shape: (num_queries, num_keys)
- attention = F.softmax(scores, dim=1)  #dim =1 => softmax each rows

## Scaled Dot-Product Attention (From the paper)
- scores = (query @ keys.T)/math.sqrt(d_k)
- Paper divides by √d_k to prevent large scores
- Large scores -> softmax saturates -> gradients vanish
- For higher dimensions (like d_k = 512 in paper)


## Week 2: Self-Attention

### Input Representation
- Each word = embedding vector (4D, 512D, etc.)
- Sentenece = matrix of shape (seq_len, embed_dim)
- Example: "The Cat sat" = (3,4) matrix

### Creating Q,K,V from Same Input
- W_q , W_k, W_v : Linear Transformation for input using NN(matmul + bias)
- Q, K, V : Linear Transformation with the respective weights
- All have same shape as INPUT

### Self Attention steps
# Create Q,K,V
- Q,K,V = W_q(x), W_k(x), W_v(x)
# Scaled dot-product attention
- d_k = K.shape[-1] for dimension of vector
- Scores computed with Scaled Dot Product 
- Attention computed using softmax on scores' last dimension(rows)
# Output
- output = attention @ V

**Shapes:**
- Input: (seq_len, embed_dim)
- Q, K, V: (seq_len, embed_dim)
- Scores: (seq_len, seq_len) ← Attention matrix!
- Output: (seq_len, embed_dim) ← Same as input

### Self-Attention reusable Module
- A class module SelfAttention to work on single input or batch input
# Basic Python OOP
- nn.Module  -> Parent Class for inheritance for NN in PyTorch
- __init()__ -> Constructor with 'embed_dim' parameter
- super()    -> To use parent class method, so that it can be tracked by Pytorch for internal bookkeeping of layers
-  self      -> To specify the current instance
- forward()  -> Special method in PyTorch. When you write self_attn(x), Python calls forward(x)  behind the scenes, ie : __call__ method

# Batch Processing
- Similar to week 1 but we have multiple sentences here rather than multiple queries
- Using last 2 dimensions to get Word Embeddings Vector
# Pattern Words
- For identical word embeddings, the attention values are identical
- Most attended word is the max attention value which comes first in traversal.

