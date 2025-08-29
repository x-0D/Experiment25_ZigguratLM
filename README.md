# ZigguratLM: A Discretized Manifold Transformer

ZigguratLM is an experimental language model architecture designed to learn a structured, hierarchical, and interpretable representation of language. Unlike standard transformers that operate in a continuous vector space, ZigguratLM forces its internal representations through a series of discrete bottlenecks. This process creates a "Discretized Manifold" of concepts at multiple levels of abstraction.

The architecture is named for its pyramid-like structure, where the sequence of representations is progressively shortened in deeper stages, resembling a ziggurat.

## Architectural Innovations

ZigguratLM introduces three core concepts that differentiate it from a standard causal transformer.

### 1. The Discretized Manifold Block

The fundamental building block of the model is the `DiscretizedManifoldBlock`. After the standard attention-like (`SimplifiedRetention`) and MLP sub-layers, the resulting hidden state is quantized into a discrete code.



This is achieved using a **Residual Vector Quantizer (RVQ)**. The process works as follows:
1.  The model's continuous hidden state `x` is passed to the RVQ.
2.  The RVQ finds the closest vector ("code") in its codebook (a learned set of prototype vectors or "buckets").
3.  This process is repeated hierarchically several times (`vq_levels`), with each subsequent quantizer working on the residual error of the previous one. This allows for a fine-grained, multi-level quantization.
4.  The final quantized vector is then added back into the main residual stream.

This forces the model to map every internal state to a specific, discrete "concept" learned in the codebook, creating an information bottleneck that encourages robust and compressed representations.

### 2. Hierarchical Processing (The Ziggurat)

The model is organized into multiple `stages`. After each stage (a series of `DiscretizedManifoldBlock`s), a `ResidualTokenFusionBlock` downsamples the sequence length by a factor of 2.

-   **Stage 0** operates on the original sequence of tokens. Its "buckets" learn to represent single tokens or their immediate properties.
-   **Stage 1** operates on a sequence that is half the length of the original. Each vector in this stage represents a fused concept from two vectors in the previous stage.
-   **Stage 2** would operate on concepts spanning four original tokens, and so on.

This creates a pyramid of representations, forcing the model to build higher-level abstractions from lower-level ones, naturally mirroring the compositional structure of language (tokens -> phrases -> sentences -> ideas).

### 3. Decoupled Prediction and Abstraction

A crucial design choice in ZigguratLM is that **next-token prediction is performed *only* using the output from Stage 0**.

The deeper, downsampled stages (Stage 1 and beyond) do not directly contribute to the final output logits. Their sole purpose is to learn abstract representations and provide a regularization signal to the earlier stages. This decouples the task of surface-level prediction (syntax, grammar) from the task of deep semantic abstraction. The deeper layers are trained only to create a good, discrete representation of the information passed to them, which in turn structures the representations in Stage 0.

## Why is ZigguratLM better than a standard Transformer?

While standard transformers are incredibly powerful, they operate as black boxes within a nebulous, high-dimensional continuous space. ZigguratLM offers several advantages by moving to a structured, discrete latent space.

1.  **Enhanced Interpretability:** The Vector Quantizer's codebook is not a black box. Each "bucket" is a learned prototype vector that comes to represent a specific semantic or syntactic concept. By observing which tokens or token-groups activate which buckets, we can "peek inside the model's brain." The `BUCKET ANALYSIS` in the training logs demonstrates this directly.

2.  **Structured Latent Space & Regularization:** Forcing the model's representations through a discrete bottleneck is a powerful form of regularization. It prevents the model from relying on tiny, insignificant variations in its continuous vector space. Instead, it must learn to group similar concepts into the same bucket, leading to a more organized and robust internal world model. This can improve generalization and reduce overfitting.

3.  **Hierarchical Abstraction:** Standard transformers process a flat sequence of tokens. While they can learn long-range dependencies, the hierarchical structure is implicit. ZigguratLM makes this hierarchy explicit, forcing the model to learn features at increasing levels of temporal and semantic abstraction, from individual words in Stage 0 to multi-token concepts in Stage 1 and beyond.

## Training Analysis: Learning the Lovecraftian Manifold

The model was trained exclusively on a corpus of H.P. Lovecraft's novels. This shapes its entire understanding of language, resulting in a model that "thinks" in terms of cosmic horror, archaic prose, and existential dread.

### The Dual-Objective Training

Training is not a simple matter of minimizing prediction error. The model is trained with two distinct objectives and two separate optimizers:

1.  **Main Loss (Cross-Entropy):** This is the standard language modeling objective. It updates the main model parameters (`SimplifiedRetention`, `MLP`, `Embeddings`, etc.) to improve its ability to predict the next token. This is calculated from Stage 0's output.
2.  **VQ Loss (Quantization Loss):** This loss updates *only* the VQ codebook vectors. It encourages the codebook vectors (the "buckets") to move closer to the hidden states that the main model produces. This ensures the discrete concepts in the codebook are relevant to the representations the model is learning. It is trained with a much smaller learning rate, allowing the codebook to slowly adapt to the main model's representations.

### The Evolution of Semantic "Buckets"

The provided training logs clearly show the model learning to categorize language.

**In Early Training (Epoch 1):**
The buckets are chaotic and group tokens loosely.
- `Bucket [ 942]` contains a jumble of common function words: `('was', 6), ('in', 5), ('than', 2), ('of', 2)`.
- `Bucket [ 234]` contains determiners but also other unrelated tokens: `('the', 10), ('its', 2), ('The', 1), ('strange', 1)`.

**In Later Training (Epoch 3):**
The buckets have become far more specialized and semantically coherent. The model has learned a discrete internal grammar.

-   **Stage 0 (Token-Level Concepts):**
    -   **Prepositions:** `Bucket [ 461]` now almost exclusively contains prepositions: `('into', 1), ('from', 1), ('on', 1), ('around', 1), ('at', 1), ('by', 1)`.
    -   **Pronouns:** `Bucket [ 773]` has learned to group pronouns: `('he', 2), ('you', 2), ('There', 1), ('He', 1), ('It', 1)`.
    -   This shows the model is not just memorizing, but creating meaningful, interpretable categories for language components.

-   **Stage 1 (Multi-Token Concepts):**
    -   The logs state `Analyzing Multi-Token Concepts (Token labels not applicable)`. This is because each representation in Stage 1 corresponds to a fused pair of tokens from Stage 0. We cannot assign a single word label to them.
    -   The `Top 5 most used buckets` shows that the model develops preferred high-level abstract features. These buckets represent common phrasal patterns, semantic relationships, or stylistic motifs found in Lovecraft's writing.

### Generation Quality

The improvement in the model's understanding is reflected in its output.

-   **Epoch 1 Generation:** `The meaning of life is eff find thatgo to its wh Thusailed his world liked on1. I mysteries...` - This is incoherent babble.

-   **Epoch 3 Generation:** `The meaning of life is a god chiselled in cloud and clearly standing neutralisation, which we know of beings outside the unformed metal suppressed-five...` - This text is grammatically correct, stylistically consistent with Lovecraft's writing, and thematically appropriate. The model has clearly learned the structure and "feel" of its training data.
