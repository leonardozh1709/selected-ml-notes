
A standard Transformer MLP layer (ignoring bias terms for simplicity) takes an input vector $\mathbf{x}$ from the residual stream and performs two linear transformations with a non-linear activation $\sigma$ (like GELU):

$$\text{MLP}(\mathbf{x}) = \sigma(\mathbf{x} W_1) W_2$$

If we think of $W_1$ as matrix of key vectors $[k_1, ..., k_d]$, the Dot Product $\mathbf{x} \cdot \mathbf{k}_i$ (which happens during the Matmul) measures the unnormalized cosine similarity (directional alignment scaled by magnitude) of the current token state $\mathbf{x}$ against key vector $\mathbf{k}_i$: This product is bigger if the vectors **point in similar directions** and is also scaled by their magnitude.

- **Key Vector $\mathbf{k}_i$:** Detects a specific semantic pattern or condition, **a fact** (e.g., _"the current token is something edible"_).
    
- **Dot Product + Activation $\sigma(\cdot)$:** Returns a high positive score if $\mathbf{x}$ matches the pattern $\mathbf{k}_i$, and near-zero otherwise. *Functions like GELU or ReLU act as a **gate/threshold** that suppress non-matches to near-zero*.

If we think of $W_2$ as a matrix of value vectors $[v_1, ..., v_d]$, then we can say that if the key matches, the non-linear activation turns "ON" neuron $i$, multiplying $\sigma(xW_1)$ by value vector $\mathbf{v}_i$.

- **Value Vector $\mathbf{v}_i$:** Contains factual or linguistic updates associated with that key (e.g., boosting the vocabulary probability for the token `"Apple"` or `"Banana"`).
    
- **Residual Addition:** The retrieved value $\mathbf{v}_i$ is written directly back into the token’s residual stream.

---
### Geometric Interpretation of "Facts"

We can think of mutually exclusive facts such as *"x is a fruit"* and *"x is a vehicle"* as value vectors stored in the linear layer weight matrices that are **almost right angled** to each other, this means that their dot product is almost zero ($\cos(90)=0$). 

Because high-dimensional spaces allow for a massive number of nearly-orthogonal vectors, a Transformer can have thousands of distinct concept directions in one lower-dimensional latent space without them interfering with one another.

>[!EXAMPLE] Example in 1024 dimensions
>If you require **perfect orthogonality, exactly 1,024 vectors** can fit. However, if you allow near-orthagonality ($81.4°$ to $98.6°$), you can fit over 100'000 of those vector in the same latent space. This happens because high-dimensional space expands exponentially.
>$\rightarrow$ This allows LLMs to store millions of concepts.

---
### Interaction with Self-Attention

During Self-Attention the tokens **aggregate information from each other** and during the feedforward blocks the **"think" and process that information**. Stacking these layers after each other allows the network to learn semantically abstract (increasingly high-level) connections and facts.

>[!IMPORTANT] **The "Thinking" Engine:**
>
>- **Attention (Information Router - Across Sequence Dimension):** Aggregates context across sequence positions (routes _where_ info goes).
  >  
>- **MLP (Processing & Memory - Across Channel Dimension (spatially independent)):** Acts as key-value lookup memory and non-linear processing to transform features into facts and logical outputs.

| Memory Type         | Keys and Values                                                                            | Where does the Information come from?                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Attention Key-Value | Dynamic representations derived from **other tokens in the prompt**. (Linear combinations) | **Context / Sequence** (Short-term memory) or additional signal (Cross-Attention) |
| MLP Key-Value       | Static weight vectors $\mathbf{k}_i$ and $\mathbf{v}_i$ learned during **pre-training**.   | **Model Parameters** (Long-term factual memory)                                                                  |

---
### The Residual Stream as Communication

The residual stream $\mathbf{x}_l$ is a shared hyper-dimensional "memory bus" running down the entire depth of the network.

A single encoder layer is defined as:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{Attention}(\mathbf{x}_l) + \text{MLP}(\mathbf{x}_l)$$

Attention and MLP layers read from the stream via projection (*geometrically, this means they convert into a different high-dimensional space*), perform their task, and **add write-updates directly back into the stream**. They do not erase past state. They accumulate information.

---
### The Transformer as a Universal Function Approximator

The synergy between the dynamic context aggregation in attention and information processing & storage is a big part of what allows Transformers to be insanely scalable and useful across a wide range of tasks. (Sequence Modelling, Generative Modelling such as FM and Diffusion, Computer Vision).

The [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) applies to feedforward networks with a single hidden layer with non-polynomial activation functions. It states that Neural Networks with a certain structure can theoretically approximate any continuous function to arbitrary accuracy.

>[!INFO]
>This only guarantees that such a network **exists**. It does not provide a method for finding the networks parameters and they don't specify how large the network must be.

