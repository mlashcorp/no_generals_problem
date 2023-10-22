---
title: "LLM Distributed Inference for Fun and not Profit - Part 1"
date: 2023-10-21T16:31:22+01:00
draft: false
math: katex
---

![distributed inference](ngp_1.jpeg)

Recently, I got very interested in learning and understanding how LLMs work. Specifically, I was curious about how LLMs are used at scale (i.e., with multiple nodes). 

One of the best resources I found to learn about LLMs was Andrej Karpathy's YouTube video, where he [created GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
.

> Also, if, like me, your fundamentals on neural networks are a bit rusty, he has a [whole series](https://karpathy.ai/zero-to-hero.html) where he builds up from the basic concepts of neural nets up to the transformer model.

—

Having learned the basics of how a Transformer model works, it was time to dive into the gory details of distributing the computation of an LLM. 

My intuition told me to start with the problem of distributing the forward pass (inference, in which we provide input to an LLM and get generated text as a result), if nothing else, because it felt like an easier problem to tackle compared to distributed training.

Most of the work I've seen focuses on Training parallelism - using multiple machines to accelerate the process of training the transformer model. However, training a neural network requires both forward and backward passes, so we should be able to leverage existing techniques used for training to our goal of distributing inference. 

To me, it makes intuitive sense that most information about LLM parallelism is related to training. This is a non-interactive operation (not sensitive to latency) that is quite compute-intensive. LLM Inference, on the other hand, is typically used in applications where the time to response is critical (such as chatGPT) - in a distributed setting, unless the problem we are tackling is [embarrassingly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel), we can expect the need to synchronize data between nodes, which will add latency to our application. However, I believe there are still several applications that would benefit from distributing inference at the cost of latency (such as batch operations that require LLM capabilities), and I'm also keen to explore how fast I can perform inference using multiple nodes.

In this blog series, I will explore existing algorithms to perform distributed inference for LLMs, their limitations and tradeoffs, and (try to) implement them from scratch to understand them better.

## How many ways can you split a matrix?

Naturally, there already exists plenty of work dedicated to distributing the training of LLMs, and as I said, many of the concepts and techniques developed for parallel training should be applicable to inference as well. OpenAI provides a [great starting point](https://openai.com/research/techniques-for-training-large-neural-networks) to understand how researchers in the field have tackled this problem. 

There are several techniques to perform parallel training of LLMs, but for the sake of simplicity, I will start by dividing the problem into 3 options:

1. Data Parallelism - where we load the entire model in each node but only pass part of the training data. Gradients are then averaged across workers. This option is not applicable for inference since LLMs are [auto-regressive](https://www.investopedia.com/terms/a/autoregressive.asp#:~:text=A%20statistical%20model%20is%20autoregressive,based%20on%20its%20past%20performance.) and, as such, require previously generated tokens to predict a given token.



2. Pipeline Parallelism - here, we load a subset of the model's layers in each node and sequentially pass activations until we reach the network's end. Intuitively, because there is a sequential dependency between nodes, we can expect that there will be no gains from concurrently processing data in multiple machines. However, we can split a large model that would typically not fit in memory in multiple machines since the parameter size per node should be 1/n for n nodes.

3. Tensor Parallelism - in tensor parallelism, we leverage the fact that matrix multiplication is a problem that is trivially parallelizable. Each node maintains all layers of the network, but for each layer, only a fraction of the parameters. In this configuration, we can process each layer in parallel in multiple nodes, reducing the memory usage and the compute requirements in each node.

There are other ways to distribute the computation of LLMs, but for now, I will focus first on understanding tensor parallelism, specifically the [MegatronLM paper from Nvidia](https://arxiv.org/abs/1909.08053).


## Tensor Parallelism with MegatronLM

One of the key ideas of the MegatronLM paper is that we can leverage the mathematical properties of matrix multiplication to distribute our computation. The paper focuses on applying their ideas to the transformer model. Specifically, we will be applying these ideas to the GPT-2 architecture, which is a decoder-only transformer. 

To better understand the paper, I implemented some ideas using Andrej's GPT-2 nanoGPT implementation as a reference. The code can be found [here](https://github.com/mlashcorp/distributed-inference), and I will reference it as I go through the paper.

—


Decoder transformers are composed of several blocks of self-attention and fully connected layers. We will focus on section 3 of the paper, where the authors provide an implementation both for the fully connected layer (MLP) and for the attention layer.

![decoder](decoder.png#center)

### Let's start with the MLP

A multilayer perceptron is a kind of neural network. It is a fully connected network with a non-linear activation function. In GPT-2, this MLP has 2 linear layers with a GeLU non-linearity in between them. 

```
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

After applying the non-linearity, each node will have a resulting 1x2 matrix (lime green and brown in node 1, dark red and magenta in node 2) that contains the result in the correct format, but to calculate the final result of the MLP block, we must take both matrices from the 2 nodes and add them (using an [all_reduce operation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)) to get the final result.


