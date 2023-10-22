---
title: "LLM Distributed Inference for Fun and not Profit - Part 1"
date: 2023-10-21T16:31:22+01:00
draft: false
math: katex
---

![distributed inference](ngp_1.jpeg)

Recently, I became very interested in learning and understanding how LLMs work. Specifically, I was curious about how LLMs are used at scale (i.e., with multiple nodes). 

One of the best resources I found to learn about LLMs was Andrej Karpathy's YouTube video, where he [created GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
.

> Also, if, like me, your fundamentals on neural networks are a bit rusty, he has a [whole series](https://karpathy.ai/zero-to-hero.html) where he builds up from the basic concepts of neural nets up to the transformer model.

—

Having learned the basics of how a Transformer model works, it was time to dive into the gory details of distributing the computation of an LLM. 

My intuition told me to start with the problem of distributing the forward pass (inference, in which we provide input to an LLM and get generated text as a result), if nothing else, because it felt like an easier problem to tackle compared to distributed training.

Most of the work I've seen focuses on Training parallelism - using multiple machines to accelerate the process of training the transformer model. However, training a neural network requires both forward and backward passes, so we should be able to leverage existing techniques used for training for my goal of distributing inference. 

To me, it makes intuitive sense that most information about LLM parallelism is related to training. This is a non-interactive operation (not sensitive to latency) that is quite compute-intensive. LLM inference, on the other hand, is typically used in applications where the time to response is critical (such as chatGPT). In a distributed setting, unless the problem we are tackling is [embarrassingly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel), we can expect that the need to synchronize data between nodes will add latency to our application. However, I believe there are still several applications that would benefit from distributing inference at the cost of latency (such as batch operations that require LLM capabilities), and I'm also keen to explore how fast inference can be performed using multiple nodes.

In this blog series, I will explore existing algorithms to perform distributed inference for LLMs, their limitations and tradeoffs, and (try to) implement them from scratch to understand them better.

## How many ways can you split a matrix?

Naturally, there already exists plenty of work dedicated to distributing the training of LLMs, and as I said, many of the concepts and techniques developed for parallel training should be applicable to inference as well. OpenAI provides a [great starting point](https://openai.com/research/techniques-for-training-large-neural-networks) to understand how researchers in the field have tackled this problem. 

There are several techniques to perform parallel training of LLMs, but for the sake of simplicity, I will start by considering three popular options:

1. Data Parallelism - where we load the entire model in each node but only give each node a part of the training data. Gradients are then averaged across workers. This option is not applicable for inference since LLMs are [auto-regressive](https://www.investopedia.com/terms/a/autoregressive.asp#:~:text=A%20statistical%20model%20is%20autoregressive,based%20on%20its%20past%20performance.) and, as such, require previously generated tokens to predict a given token.



2. Pipeline Parallelism - here, we load a subset of the model's layers in each node and sequentially pass activations until we reach the network's end. Intuitively, because there is a sequential dependency between nodes, we can expect that there will be no gains from concurrently processing data in multiple machines. However, we can split a large model that would typically not fit in memory in multiple machines since the parameter size per node should be 1/n for n nodes.

3. Tensor Parallelism - which leverages the fact that matrix multiplication is a problem that is trivially parallelizable. Each node maintains all layers of the network, but for each layer, only a fraction of the parameters. In this configuration, we can process each layer in parallel in multiple nodes, reducing the memory usage and the compute requirements in each node.

There are other ways to distribute the computation of LLMs, but for now, I will focus first on understanding tensor parallelism, specifically the [MegatronLM paper from Nvidia](https://arxiv.org/abs/1909.08053).


## Tensor Parallelism with MegatronLM

One of the key ideas of the MegatronLM paper is that we can leverage the mathematical properties of matrix multiplication to distribute our computation. The paper focuses on applying their ideas to the transformer model. Specifically, we will be applying these ideas to the GPT-2 architecture, which is a decoder-only transformer. 

To better understand the paper, I implemented some of these ideas using Andrej's GPT-2 nanoGPT implementation as a reference. The code can be found [here](https://github.com/mlashcorp/distributed-inference), and I will reference it as I go through the paper.

—


Decoder transformers are composed of several blocks of self-attention and fully connected layers. We will focus on section 3 of the paper, where the authors provide an implementation both for the fully connected layer (MLP) and for the attention layer.

![decoder](decoder.png#center)

### Parallelizing the MLP

A multilayer perceptron is a kind of neural network. It is a fully connected network with a non-linear activation function. In GPT-2, this MLP has 2 linear layers with a GeLU non-linearity in between them. 

```py
def forward(self, x):
    _, T, _ = x.size() 
    x = self.c_fc(x)   # linear layer
    x = self.gelu(x)   # activation
    x = self.c_proj(x) # linear layer
```

The key idea from the paper is that we can split the parameters of each linear layer in this MLP across nodes, but how we split the parameters and reconcile the results is worth explaining.

The following illustration shows a simple representation of a multiplication of an input 1x2 matrix by a 2x2 matrix, the result of which will be a 1x2 matrix.


![gemm](gemm-1.png#center)


If we split the 2x2 matrix across its rows, each node would have a 1x2 result matrix, but the values would not be correct until we summed the matrices from both nodes. If, on the other hand, we split the 2x2 matrix across its columns, as shown in the illustration above, each node will have a 1x1 matrix with the correct final result.

In the transformer MLP, we can think of the blue matrix as the input and the 2x2 matrix (in yellow and red) as the parameters of the linear layer. The number of columns represents the hidden dimension of the linear layer, and the number of rows must match the columns of our input. The GPT-2 paper actually uses 4 * n_embed for the hidden dimension, but for simplicity's sake, I'll use n_hidden = n_embed.


To calculate the output of the MLP, we need to [multiply the two matrices](https://en.wikipedia.org/wiki/Matrix_multiplication). This is where we can split the 2x2 matrix across the two columns sending each split to a different node, then send the full input matrix to both nodes, and finally, in each node, independently (and in parallel), calculate each result element (green and pink elements in the figure). 

> This approach splits the parameter matrix of the linear layer across multiple nodes, reducing the amount of memory and computation that each node must do. The entire input matrix must still be passed to all nodes, so we get sublinear distribution gains from this operation.

![fig3a](fig3a.png#center)


The diagram above represents part of Figure 3a from the paper. In it, the authors explain the key idea of first splitting the A parameter matrix by columns and the B matrix by rows. As we saw before, if we split the second element of matrix multiplication by columns, each node will have a result matrix with a subset of the columns, whereas if we split by rows, the result matrix will have the correct result shape, but we will need to reduce (synchronize) the data between all nodes to arrive at the correct result.

The authors opted for this order of operations to minimize synchronization points. A GeLU is a non-linear operation that can only be safely applied in parallel if we split the first linear layer across its columns, resulting in the equation:


$$ [Y1, Y2] = [GeLU(X A1), GeLU(X A2)] $$

Where A is the parameters of the linear layer, A1 has the first half of the columns of the matrix, and A2 is the second half. As we saw before, we can independently calculate the resulting Y matrix in different nodes. 

Had the authors split A across its rows, they would need to synchronize data before applying the non-linear activation function - since each node would have a result matrix with the correct format after the first linear layer, but with only half of the parameters having been summed.

For the second linear layer, we must partition its parameter matrix along its rows (matrices B1 and B2) so that matrix multiplication rules are preserved. 

![gemm](gemm-2.png#center)

After applying the non-linearity, each node will have a resulting 1x2 matrix (lime green and brown in node 1, dark red and magenta in node 2) that contains the result in the correct format. However, to calculate the final result of the MLP block, we must take both matrices from the 2 nodes and add them (using an [all_reduce operation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)) to get the final result.

This is the key strategy that the authors also use in the self-attention block. 

## Adapting NanoGPT to run in parallel

To apply the ideas discussed in the previous section, I adapted the [NanoGPT implementation](https://github.com/karpathy/nanoGPT) so that the MLP and the self-attention blocks can be split across several nodes. My goal was to create a minimally viable implementation, that favors readibility against completeness or sophistication. 

The entire GPT-2 implementation is still contained in a single file, distributed_model.py. The relevant files from the repo are:

```
.
├── run.py                     <- CLI application wrapper
├── distributed_inference.py   <- Launches the processes and waits for the results.
├── distributed_model.py       <- GPT-2 with tensor parallelism
├── distributed_state.py       <- Simulated distributed all reduce
└── download_model.py          <- Use this to download the GPT-2 124M model from HF
```

A key simplification done in this codebase was the introduction of a distributed state module. This module implements a naive all_reduce operation, using Python's [BaseManager process](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.BaseManager) as a synchronization point between processes. When you run this code using multiple processes, each time a worker calls the all_reduce function, it will publish their intermediate results to the base manager, and only when all workers have submited their results, they all get the final, reduced version of the matrix:

```py
def all_reduce(self, op_id: str, tensor: torch.Tensor):
    with self.cv:
        if op_id not in self.state:
            self.state[op_id] = (tensor, 1)
        else:
            state_tensor, count = self.state[op_id]
            self.state[op_id] = (
                torch.add(state_tensor, tensor), count + 1)

        if self.state[op_id][1] < self.number_of_workers:
            self.cv.wait()
        else:
            self.cv.notify_all()
        return self.state[op_id][0]
```

For each operation (uniquely identified using the op_id - using the worker index, the block name, and the current token index) we sum the input matrix with the existing state, and block the caller on a condition variable. When the final worker calls the all_reduce function, we notify all blocked users and return the final matrix.

> The points where we use all reduce are the synchronization points between machines. The amount of data we need to transfer and the inter-node latency will be the bottlenecks for the forward pass performance.

This first version of the code does not address all aspects presented in the paper. I simply focused on implementing the distributed MLP and self-attention blocks. Other aspects such as word, position and transposed embeddings were not distributed yet.

There are two key areas worth exploring in this code. 1) how I'm loading the model shard in each node; and 2) how I'm setting the MLP and Self-Attention parameter sizes. Let's look at model loading first.

### Sharded model loading

Model loading is done in the load_layers function of the distributed_mode.py module. I leverage safetensors to load a slice of each layer selectively. As an example:

```py
if "mlp.c_fc.weight" in layer:
    # Partition by column. This will convert the slice to a Tensor object
    tensor_slice = f.get_slice(layer)
    tensors[layer] = tensor_slice[:, mlp_start_idx:mlp_end_idx]
```
Here I'm loading a slice of the first linear layer of the MLP block by loading only the columns allocated to this worker node. Start and end indeces for the columns are calculated using this helper function:

```py
        def get_worker_partition(C: int = 768,
                                 worker_index: int = 0,
                                 number_of_workers: int = 1):
            partition_size = C // number_of_workers
            # Calculate the "chunk" that this node will process
            partition_start = worker_index * partition_size
            partition_end = partition_start + partition_size
            return (partition_start, partition_end)
```

Then, in the MLP definition, I change the shape of the linear layers to match the way we are loading the model. If we recall the paper, we must partition the first linear layer across its columns, and the second across its rows (by dividing by config.number_of_workers):

```py
    self.c_fc = nn.Linear(config.n_embd, (4 * config.n_embd) //
                            config.number_of_workers, bias=config.bias)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(
        (4 * config.n_embd) // config.number_of_workers, config.n_embd, bias=config.bias)
```

Finally, in the forward pass, we must sum the result of this operation across all nodes by using the (simulated) all_reduce operation:

```py
    def forward(self, x):
        _, T, _ = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = self.c_fc(x)    # linear layer
        x = self.gelu(x)    # activation
        x = self.c_proj(x)  # linear layer

        # All reduce the output of the MLP. (Synchronization point)
        op_id = f"{self.layer_id}_{T}"
        x = self.reduce_controller.all_reduce(op_id, x)

        x = self.dropout(x)
        return x
```

The same idea is applied to the self-attention block, and I invite you to read the code and try it out. 


That's all for today, next I will finish implementing the paper by distributing the embeddings, and use Torch's all_reduce and run the code in different nodes. See you then!
