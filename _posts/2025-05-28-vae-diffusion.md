---
layout: post
title: "From VAEs to Diffusion Models: A Step-by-Step Journey"
subtitle: "Understanding the evolution of generative models through practical implementations"
cover-img: /assets/img/posts/2025-05-28/vae_diffusion.webp
thumbnail-img: /assets/img/posts/2025-05-28/vae_diffusion.webp
tags: [vae, diffusion-models, generative-ai, deep-learning, pytorch, tutorial]
author: Taewoon Kim
mathjax: true
---

## Motivation: Why Do We Need Generative Models?

In order to generate data, we need to sample from a distribution. But here, we are not
talking about simply sampling from the training data distribution — instead, we want to
**train a neural network** that can represent a distribution, and then sample *from the
network itself*.

There are two dominant paradigms for this:

1. **Autoregressive Models**: These models generate data step-by-step, predicting the
   next "token" (e.g., a word or a pixel) given the previous ones. They are especially
   powerful for **discrete data** like text, where we use **categorical distributions**
   to model the next item in a sequence.

2. **Latent Variable Models**: These include **VAEs** and **Diffusion Models**, where we
   introduce latent variables (usually continuous) and train a network to sample from
   them. This approach is particularly effective for **continuous data** like images,
   where **Gaussian distributions** naturally model pixel values and latent  
   representations.

These two paradigms are not mutually exclusive — there are autoregressive models for
images and latent variable models for text. However, in this post, we focus on the
**most classical examples** of latent variable models — **VAEs** and **Diffusion
Models** — to demonstrate how a neural network can learn to generate pixel images from
"noise".

This journey will follow three practical stages:

- [**VAEs**](https://github.com/tae898/vae-diffusion/blob/main/01.vae.ipynb): The
  foundational latent variable model
- [**VAEs without
  Encoders**](https://github.com/tae898/vae-diffusion/blob/main/02.vae-without-encoder.ipynb):
  A simplified one-step model bridging toward diffusion
- [**Diffusion
  Models**](https://github.com/tae898/vae-diffusion/blob/main/03.diffusion.ipynb): The
  multi-step denoising approach

By understanding this progression, you'll gain insights into why diffusion models have
become so powerful and how they relate to their predecessors.

---

## 1. Variational Autoencoders (VAEs)

### 1.1 Motivation: Learning to Generate Images

VAEs aim to learn a generative model of data by introducing **latent variables** that
capture the underlying structure. Formally, we want to model:

$$ p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz $$

Where:
- $$x$$ is our data (e.g., images)  
- $$z$$ is a latent variable (typically lower-dimensional)  
- $$p(z)$$ is a simple prior (usually $$\mathcal{N}(0, I)$$)  
- $$p_\theta(x \mid z)$$ is a neural network "decoder"  

The challenge is that this integral is **intractable** because the decoder is a complex
neural network. This is where variational inference comes in.

### 1.2 The ELBO: Training through Variational Inference

To make training feasible, VAEs introduce an **approximate posterior** (encoder)
$$q_\phi(z \mid x)$$ and optimize the **Evidence Lower BOund (ELBO)**:

$$ \log p_\theta(x) \;\ge\; \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x \mid
z)\bigr] \;-\; D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\,p(z)\bigr) $$

This bound consists of:
- A **reconstruction term** that ensures the decoded samples look like the input
- A **KL divergence term** that regularizes the latent space toward the prior

### 1.3 Implementation: Encoders and Decoders

Our VAE implementation has two key components:

1. **Encoder** $$q_\phi(z \mid x)$$: Maps inputs to latent distribution parameters (mean
   and variance)  
2. **Decoder** $$p_\theta(x \mid z)$$: Reconstructs inputs from latent samples

For training, we use the **reparameterization trick** to enable gradient flow:

$$ z = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I) $$

#### The Reparameterization Trick: A Closer Look

The reparameterization trick is crucial for training VAEs with gradient descent. When
sampling from a distribution (like our encoder's output distribution), the operation is
non-differentiable, which breaks the computational graph needed for backpropagation.

To solve this, we express sampling as a deterministic function of the distribution
parameters and an external source of randomness:

1. **Without reparameterization**: $$z \sim q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x),
   \sigma^2_\phi(x))$$
   - This direct sampling breaks gradient flow from $$z$$ back to $$\phi$$.

2. **With reparameterization**: $$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon$$,
   where $$\epsilon \sim \mathcal{N}(0,I)$$
   - The randomness is externalized in $$\epsilon$$
   - We get the same distribution for $$z$$
   - The gradient can flow from $$z$$ back to $$\phi$$

This technique allows us to optimize the encoder through standard backpropagation, even
though we're sampling from its output distribution.

### 1.4 Two Decoder Variants: Bernoulli vs Gaussian

As discussed in my previous post on Maximum Likelihood Estimation (MLE) [Understanding
Maximum Likelihood Estimation](https://taewoon.kim/2025-02-05-mle/), different
likelihood assumptions correspond to different loss functions. Here, we explore
Bernoulli and Gaussian likelihoods for the decoder:

**Bernoulli Likelihood**:  
$$ p_\theta(x \mid z) = \prod_{i=1}^D \mathrm{Bernoulli}(x_i;\,\hat x_i) $$ This gives
the **binary cross-entropy** loss: $$ \log p_\theta(x \mid z) =
\sum_{i=1}^D\bigl[x_i\log\hat x_i + (1-x_i)\log(1-\hat x_i)\bigr] $$

**Gaussian Likelihood**:  
$$ p_\theta(x \mid z) = \mathcal{N}(x;\,\hat x,I) $$ This simplifies to the **mean
squared error**: $$ \log p_\theta(x \mid z) = -\tfrac12\|x-\hat x\|^2 + \text{const} $$

### 1.5 Architecture Comparison: MLP vs CNN

2D CNNs were desgined to process spatial data. Of course we can just turn a 2D matrix
into a 1D vector and use an MLP, but this doesn't help the neural network to know the
spatial structure of the input data.

---

## 2. VAE Without Encoder: A Bridge to Diffusion

### 2.1 Simplifying the Model: Fixed Corruption Process

What if, instead of learning an encoder, we use a **fixed corruption process**?

$$ x_1 = \sqrt{\alpha}\,x_0 \;+\;\sqrt{1-\alpha}\,\epsilon,\quad
\epsilon\sim\mathcal{N}(0,I) $$

This creates a "latent" $$x_1$$ by adding controlled Gaussian noise to the original
image $$x_0$$.

### 2.2 The Simplified ELBO

We can still derive an ELBO, but now with a fixed "encoder" $$q(x_1 \mid x_0)$$:

$$ \mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q(x_1 \mid x_0)}\bigl[\log p_\theta(x_0
\mid x_1)\bigr] \;-\; D_{\mathrm{KL}}\bigl(q(x_1 \mid x_0)\,\|\,p(x_1)\bigr) $$

The KL term becomes a constant w.r.t. the model parameters, so training focuses on the
reconstruction term.

### 2.3 From Reconstruction to Noise Prediction

A key insight: instead of directly predicting $$x_0$$ from $$x_1$$, we predict the
**noise** $$\epsilon$$ that was added:

$$ \epsilon \approx \epsilon_\theta(x_1) $$

This yields the **noise prediction objective**:

$$ \mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{x_0,\epsilon}\bigl\|\epsilon -
\epsilon_\theta(x_1)\bigr\|^2 $$

To understand why this works, recall our forward process:

$$ x_1 = \sqrt{\alpha}\,x_0 \;+\;\sqrt{1-\alpha}\,\epsilon,\quad
\epsilon\sim\mathcal{N}(0,I) $$

We can solve for $$x_0$$ in terms of $$x_1$$ and $$\epsilon$$:

$$ x_0 = \frac{x_1 - \sqrt{1-\alpha}\,\epsilon}{\sqrt{\alpha}} $$

Now, instead of our model predicting $$x_0$$ directly, we train it to predict the noise
$$\epsilon$$ that was added. Let's denote our model's noise prediction as
$$\hat{\epsilon} = \epsilon_\theta(x_1)$$.

We can use this predicted noise to estimate $$x_0$$:

$$ \hat{x}_0 = \frac{x_1 - \sqrt{1-\alpha}\,\hat{\epsilon}}{\sqrt{\alpha}} $$

If we assume our decoder models a Gaussian likelihood $$p_\theta(x_0 \mid x_1) =
\mathcal{N}(x_0; \mu_\theta(x_1), \sigma^2 I)$$, then maximizing the log-likelihood
means minimizing:

$$ \|\hat{x}_0 - x_0\|^2 $$

Substituting our expressions:

$$ \begin{aligned} \|\hat{x}_0 - x_0\|^2 &= \left\|\frac{x_1 -
\sqrt{1-\alpha}\,\hat{\epsilon}}{\sqrt{\alpha}} - \frac{x_1 -
\sqrt{1-\alpha}\,\epsilon}{\sqrt{\alpha}}\right\|^2 \\
&= \frac{1}{\alpha}\left\|\sqrt{1-\alpha}(\epsilon - \hat{\epsilon})\right\|^2 \\
&= \frac{1-\alpha}{\alpha}\left\|\epsilon - \hat{\epsilon}\right\|^2 \end{aligned} $$

This is proportional to $$\|\epsilon - \hat{\epsilon}\|^2$$, which is our noise
prediction loss. Since $$\frac{1-\alpha}{\alpha}$$ is constant with respect to our model
parameters, we can simplify to:

$$ \mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{x_0,\epsilon}\bigl\|\epsilon -
\epsilon_\theta(x_1)\bigr\|^2 $$

Thus, predicting the noise $$\epsilon$$ is mathematically equivalent to predicting
$$x_0$$, but often leads to more stable training.

### 2.4 Limitations of One-Step Models

One-step denoising proves challenging:

1. **Jumping from noise to signal** in a single step is hard — errors compound.
2. **Signal vs. prior trade-off**:
   - High $$\alpha$$ → more signal, but $$x_1$$ deviates from a true Gaussian prior.
   - Low $$\alpha$$ → matches Gaussian prior, but little signal remains.
3. Although the U-net can better generate realistic images than the simple CNN, it still
   struggles.

#### Experimental Evidence

In our notebook experiments, we tested both a simple CNN and a U-Net architecture across
various values of $$\alpha$$ (the noise control parameter):

- With $$\alpha = 1.0$$ (no noise):
  - The model sees perfect inputs but must predict zero noise
  - Training becomes unstable or trivial
  - No generalization possible

- With $$\alpha = 0.9$$ (low noise):
  - The model sees $$x_1 \approx 0.95 \, x_0 + 0.31 \, \epsilon$$
  - Very little corruption, easy to memorize inputs

- With $$\alpha = 0.5$$ (moderate noise):
  - The model sees $$x_1 \approx 0.707 \, x_0 + 0.707 \, \epsilon$$
  - Half signal, half noise

- With $$\alpha = 0.1$$ (high noise):
  - The model sees $$x_1 \approx 0.316 \, x_0 + 0.949 \, \epsilon$$
  - Strong corruption, mostly noise
  - U-Net begins to generate recognizable digits, showing its power in denoising

- With $$\alpha = 0.0$$ (pure noise):
  - The model sees $$x_1 = \epsilon$$ (pure noise)
  - No signal at all from the original image
  - Both architectures fail completely

This demonstrates a fundamental tension: **architecture matters significantly**. The
U-Net's skip connections and multi-scale processing allow it to extract meaningful
signal from heavy noise, while simple CNNs cannot.

Moreover, there's an inherent trade-off: for the corrupted input $$x_1 = \sqrt{\alpha}
x_0 + \sqrt{1 - \alpha} \epsilon$$, the distribution only resembles a standard Gaussian
$$\mathcal{N}(0, I)$$ when $$\alpha$$ approaches 0, but then the model sees almost no
signal from $$x_0$$.

So in our notebook, we introduced a "hack", where we didn't sample directly from a
standard Gaussian, but instead from the empirical distribution of $$x_1$$ with
precomputed mean and standard deviation. However, this approach is somewhat ad-hoc and
doesn't guarantee that $$x_1$$ follows a true Gaussian distribution.

These challenges explain why single-step models struggle, and why **multi-step
diffusion** offers a more effective approach by breaking the problem into many simpler
denoising steps.

---

## 3. Diffusion Models: The Multi-Step Solution

### 3.1 Forward Process: Gradual Noising

Diffusion models add noise over $$T$$ steps:

$$ q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t
I\bigr) $$

A closed-form sample from $$x_0$$ is

$$ x_t = \sqrt{\bar\alpha_t}\,x_0 \;+\;\sqrt{1-\bar\alpha_t}\,\epsilon,\quad
\bar\alpha_t = \prod_{i=1}^t(1-\beta_i),\;\epsilon\sim\mathcal{N}(0,I). $$

#### Beta Schedules: The Key to Controlled Noise Addition

The choice of $$\beta_t$$ schedule significantly impacts model performance:

**Linear schedule** is the most straightforward one, where $$\beta_t$$ increases
linearly from $$\beta_{\text{min}}$$ to $$\beta_{\text{max}}$$. It's simple but can lead
to either too little noise in early steps or too much in later steps

The schedule directly controls how quickly information about $$x_0$$ is lost during the
forward process, which affects how difficult each denoising step will be during
generation.

### 3.2 Reverse Process: Learning to Denoise

We learn $$p_\theta(x_{t-1}\mid x_t)$$ and define

$$ p_\theta(x_{0:T}) = p(x_T)\,\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t). $$

### 3.3 ELBO for Diffusion Models

The ELBO decomposes as

$$
\mathcal{L}_{\text{ELBO}} =
\underbrace{\mathbb{E}_{q(x_1 \mid x_0)} \left[ \log p_\theta(x_0 \mid x_1) \right]}_{\text{Reconstruction}}
- \sum_{t=2}^T\mathbb{E}_{q(x_t,x_{t-1}\mid x_0)}
\underbrace{D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{\text{Reverse KLs}}
- \underbrace{D_{\text{KL}}(q(x_T \mid x_0) \,\|\, p(x_T))}_{\text{Prior term}}
$$

This simplifies to the same **noise prediction** objective:

$$
\boxed{
\mathcal{L}_{\text{diffusion}} =
\mathbb{E}_{x_0 \sim p_{\text{data}}} \;
\mathbb{E}_{t \sim \text{Uniform}(1, T)} \;
\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[
\left\| \epsilon - \epsilon_\theta\left(x_t, t\right) \right\|^2
\right]
}
$$

where:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

See [the notebook](https://github.com/tae898/vae-diffusion/blob/main/03.diffusion.ipynb)
for the detailed derivation.


### 3.4 UNet with Time Conditioning

A core component of diffusion models is the neural network architecture used for
denoising. We use a **U-Net architecture** instead of a simple CNN because:

1. **Skip connections** preserve important spatial details from early layers
2. **Downsampling and upsampling paths** allow the model to capture both fine details
   and global context
3. **Multi-scale processing** helps the model understand patterns at different
   resolutions

Because diffusion happens over multiple timesteps with varying noise levels, the model
needs to know **which timestep** it's currently denoising. We accomplish this through
**time conditioning**:

- Each timestep $$t$$ is converted into a **time embedding vector** using sinusoidal
  functions
- This vector is then incorporated into the U-Net's feature maps by:
  - Reshaping it to match the channel dimension of the current feature map
  - Simply adding it element-wise to the feature representations
  - Applying a non-linear activation function afterward

This time information is injected at multiple points in both the encoder and decoder
paths, enabling the model to adapt its denoising behavior based on the current noise
level.

This approach gives the model a "sense of time" - it knows whether it's dealing with
heavily noisy images (early in the reverse process) or nearly clean images (later in the
process), and can adjust its denoising strategy accordingly.

The sinusoidal time embedding used here is the same as the positional encoding used in
the 2017 original Transforme model, providing a unique signature for each timestep that
contains information about both absolute position and relative distances between
timesteps.


### 3.5 The Sampling Process

**Initialize**:  
$$ x_T \sim \mathcal{N}(0, I) $$

**Iterative denoising** 

for $$ t = T, \dots, 1 $$:

$$ \begin{aligned} \hat{\epsilon} &= \epsilon_\theta(x_t, t) \\
\mu_\theta(x_t, t) &= \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 -
\bar{\alpha}_t}} \, \hat{\epsilon} \right) \\
x_{t-1} &= \mu_\theta(x_t, t) + \sqrt{\beta_t} \cdot z_t, \quad z_t \sim \begin{cases}
\mathcal{N}(0, I), & \text{if } t > 1 \\
0, & \text{if } t = 1 \end{cases} \end{aligned} $$

**Rescale** to image range:  
$$ x_0^{(\mathrm{image})} = \frac{x_0 + 1}{2} \in [0, 1] $$

#### Sampling Techniques and Considerations

The sampling process involves several practical considerations:

1. **Number of sampling steps**: While training typically uses a fixed number of steps
   (e.g., T=1000), sampling can often be done with fewer steps (e.g., 50-100) with
   minimal quality loss.

2. **Deterministic vs. stochastic sampling**: For the final step (t=1), we often use
   deterministic sampling (setting z₁=0) to reduce noise in the final output.

3. **Clipping techniques**: To prevent extreme values, two approaches are common:
   - Clipping the final output to [-1,1] before rescaling
   - Estimating x₀ at each step and using it to guide the reverse process

4. **Sample quality vs. speed tradeoff**: More sampling steps generally yield higher
   quality but slower generation.

Our experiments show that with sufficient training and properly tuned hyperparameters,
diffusion models can generate remarkably realistic MNIST digits even with a
modestly-sized network.

---

## 4. Connecting the Dots: From VAEs to Diffusion

These three approaches form a clear progression:

- **VAEs**: Learn encoder $$q_\phi(z\mid x)$$ and decoder $$p_\theta(x\mid z)$$ with a
  single latent layer  
- **VAE without Encoder**: Replace the learned encoder with a fixed corruption process
  and predict noise  
- **Diffusion Models**: Extend to multiple corruption steps with a time-conditioned
  denoiser  

Key insights:

- **Noise Prediction**: Predicting noise rather than data directly improves learning
- **Multi-Step Process**: Many small denoising steps outperform a single large step
- **Time Conditioning**: Embedding "how noisy" the input is enables better denoising
- **Architectural Advances**: Each generation benefits from stronger neural designs

### The Hierarchical VAE Perspective

Diffusion models can be viewed as a special type of hierarchical VAE where:

1. The latent variables form a Markov chain: $$z_1, z_2, ..., z_T$$
2. The inference model (encoder) is fixed rather than learned
3. The generative model (decoder) is trained to reverse this fixed process

This perspective helps explain why diffusion models can generate higher-quality samples
than standard VAEs:

- They have multiple layers of latent variables
- Each step involves a simpler denoising problem
- The fixed encoder eliminates issues like posterior collapse
- The multi-step nature allows for better mode coverage

---

## 5. Conclusion and Future Directions

The journey from VAEs to diffusion models demonstrates how generative modeling has
evolved by building on core principles:

- **Variational bounds** underlie both approaches  
- Both are **latent variable models** with reconstruction and regularization

Diffusion models excel by:

- Adopting a **multi-step** denoising process  
- Focusing on **noise prediction**  
- Leveraging powerful **U-Net architectures**  

Perhaps most surprisingly, despite being derived from the complex-looking ELBO, the
final training objective for diffusion models is remarkably simple. It's essentially
just mean squared error between predicted and actual noise:

$$ \mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \epsilon}\bigl\|\epsilon -
\epsilon_\theta(x_t, t)\bigr\|^2 $$

This resembles a standard supervised learning setup with MSE loss. By removing the need
for a learned encoder, diffusion models avoid many complexities of traditional VAEs,
including the reparameterization trick. The model simply learns to denoise, step by
step, which proves to be both powerful and conceptually elegant. You'll often be
surprised by this kind of simple training objectives, e.g., next token prediction, to
build surprisingly well performing AI systems.

[**The full implementation of the code is on
GitHub**](https://github.com/tae898/vae-diffusion)

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [U-Net: Convolutional Networks for Biomedical Image
  Segmentation](https://arxiv.org/abs/1505.04597)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)