---
title: "LLM Distributed Inference for Fun and not Profit - Part 1"
date: 2023-10-21T16:31:22+01:00
draft: false
---

![distributed inference](ngp_1.jpeg)

Recently, I got very interested in learning and understanding how LLMs work. Specifically, I was curious about how LLMs are used at scale (i.e., with multiple nodes). 

One of the best resources I found to learn about LLMs was Andrej Karpathy's YouTube video, where he [creates GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
.

> Also, if, like me, your fundamentals on neural networks are a bit rusty, he has a [whole series](https://karpathy.ai/zero-to-hero.html) where he builds up from the basic concepts of neural nets up to the transformer model.

—

Having learned the basics of how a Transformer model works, it was time to dive into the gory details of distributing the computation of an LLM. 

My intuition told me to start with the problem of distributing the forward pass (inference, in which we provide input to an LLM and get generated text as a result) if nothing else because it felt like an easier problem to tackle compared to distributed training.

Most of the work I've seen focuses on Training parallelism - using multiple machines to accelerate the process of training the transformer model. However, training a neural network requires both forward and backward passes, so we should be able to leverage existing techniques used for training to our goal of distributing inference. 

To me, it makes intuitive sense that most information about LLM parallelism is related to training. This is a non-interactive operation (not sensitive to latency) that is quite compute intensive. LLM Inference on the other hand is typically used in applications where the time to response is critical (such as chatGPT) - in a distributed setting, unless the problem we are tackling is [embarrassingly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel), we can expect the need to synchronize data between nodes, which will add latency to our application. 

In this blog series, I will explore existing algorithms to perform distributed inference for LLMs, their limitations and tradeoffs, and (try to) implement them from scratch to better understand them.

## How many ways can you split a matrix?

Naturally, there already exists plenty of work dedicated to distribute the training of LLMs, and as I said, many of the concepts and techniques developed for parallel training should be applicable to inference as well. OpenAI provide a [great starting point](https://openai.com/research/techniques-for-training-large-neural-networks) to understand some of the ways researchers in the field tackled this problem. 

To simplify the topic, I will start by dividing the problem into 3 options:

1. Data Parallelism - where we load the entire model in each node, but only pass part of the training data. Gradients are then averaged across workers. This option is not applicable for inference since LLMs are auto-regressive.

2. Pipeline Parallelism - here we load a subset of the model's layers in each node, and sequentially pass activations until we reach the end of the network. Intuitively, because there is a sequential dependency between nodes, we can expect that there will be no gains from concurrently processing data in multiple machines. However, we will be able to split a large model that would typically not fit in memory in multiple machines, since the parameter size per node should be 1/n for n nodes.

3. Tensor Parallelism - in tensor parallelism we leverage the fact that matrix multiplication is problem that is trivially parallelizable. Each node maintains all layers of the network, but for each layer only a fraction of the parameters.In this configuration, we can process each layer in parallel in multiple nodes, both reducing the memory usage and the compute requirements in each node.

There are other ways to distribute the computation of LLMs, but for now I will focus first on understanding tensor parallelism, specifically, the [MegatronLM paper from Nvidia](https://arxiv.org/abs/1909.08053).


## Tensor Parallelism with MegatronLM

One of the key ideas of the MegatronLM paper is that we can leverage the mathematical properties of the matrix dot product to distribute our computation. The paper focuses on applying their ideas to the transformer model. Specifically, we will be applying these ideas to the GPT-2 architecture which is a decoder-only transfomer. 

To better understand the paper, I implemented some of its ideas using Andrej's GPT-2 nanoGPT implementation as a reference. The code can be found here, and I will reference it as I go through the paper.

—

Decoder transfomers are composed of several blocks of self-attention and fully connected layers. We will focus on section 3 of the paper where the authors provide an implementation both for the fully connected layer (MLP) and for the attention layer.

![distributed inference](decoder.png#center)

### Let's start with the MLP

A multilayer perceptron is a kind of neural network. It is a fully connected network with a non-linear activation function. In GPT-2 this MLP has 2 linear layers with a GeLU non-linearity in between them. 

```py
def forward(self, x):
    _, T, _ = x.size() 
    x = self.c_fc(x)   # linear layer
    x = self.gelu(x)   # activation
    x = self.c_proj(x) # linear layer
```

The key idea from the paper is that we can split the parameters of each linear layer in this MLP across nodes, but the way we split the parameters and the way we reconcile the results is worth explaining.