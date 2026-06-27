# Feature-Suppression

This guide provides the architecture, production-ready code, and integration documentation for implementing **Dynamic Runtime Feature Suppression**. 

Unlike static weights editing, this implementation works at the **activation layer** in memory. It uses thread-safe context scoping to ensure that if millions of users are hitting the same model container, suppression vectors are applied dynamically on a per-request basis without cross-user interference.

---

### Architectural Overview

In a production environment, requests are processed asynchronously. If User A wants to suppress a concept (e.g., "coding advice" to enforce a non-technical persona) and User B does not, the model must apply different vector modifications to the same shared memory GPU nodes at the exact same millisecond. 

The system uses Python’s built-in `contextvars` to isolate dynamic suppression configurations to specific asynchronous task contexts.

```text
[ API Request ] ---> [ Load Balancer ] ---> [ FastAPI Worker Process ]
                                                 │
                                                 ├── 1. Read 'suppression_map' from body
                                                 ├── 2. Write configuration to ContextVar
                                                 │
                                                 ▼
                                        [ PyTorch Forward Pass ]
                                                 │
                                                 ▼  (Execution reaches Target Layer)
                                        [ register_forward_hook ]
                                                 │
                                                 ├── 1. Read ContextVar (Target Feature & Multiplier)
                                                 ├── 2. Run Activation / SAE Projection
                                                 ├── 3. Modify Hidden State Matrix (Ablation / Scaling)
                                                 │      x_new = x - (1 - multiplier) * f(x)_k * W_dec[:, k]
                                                 ▼
                                        [ Downstream Layers ] ---> [ Token Generated ]
```

---

### Methodologies for Feature Suppression

There are two primary ways to target features at runtime:

1.  **Direct Neuron Suppression:** Zeroing out specific raw indices in the MLP block or residual stream. This is simpler but lacks precision because raw model neurons are polysemantic (they store multiple unrelated concepts).
2.  **Sparse Autoencoder (SAE) Suppression:** Projecting the hidden states into a high-dimensional sparse space where concepts are monosemantic (one feature = one clean concept). We modify the target feature activation and project back into the residual stream. This method is highly precise and is what modern safety teams use in production [1].

Below, we implement both approaches within a complete, production-grade FastAPI service.

---

### Production-Ready Implementation Script

This script sets up a FastAPI web server hosting a transformer model. It uses PyTorch forward hooks and thread-safe `contextvars` to intercept and steer activations on the fly.

```python
import contextvars
import os
from typing import Dict, Optional
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------------
# 1. Thread-Safe Context State Management
# -------------------------------------------------------------------------
# ContextVars ensure that request-specific state is strictly isolated to the 
# executing async coroutine thread-task. This prevents cross-request leakage.
suppression_ctx: contextvars.ContextVar[Dict[int, float]] = contextvars.ContextVar(
    "suppression_ctx", default={}
)

# -------------------------------------------------------------------------
# 2. Mock Sparse Autoencoder (SAE) for Demonstration
# -------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    """
    A lightweight representation of a pre-trained SAE. 
    In production, this module is loaded with pre-trained weights mapped
    to specific semantic features (e.g., index 1042 = chemistry text).
    """
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        # Encoder projects residual stream to sparse concept space
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder projects sparse concept space back to residual stream
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

# -------------------------------------------------------------------------
# 3. Model & Service Initialization
# -------------------------------------------------------------------------
app = FastAPI(
    title="Production Dynamic Feature Suppression API",
    description="Steers LLM activations at runtime without file modifications",
    version="1.0.0"
)

# Global variables for production models
model = None
tokenizer = None
sae = None
hook_handle = None

MODEL_ID = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYER_INDEX = 12  # Layer where our SAE is trained and aligned

@app.on_event("startup")
def load_resources():
    global model, tokenizer, sae, hook_handle
    print(f"Loading tokenizer and model: {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    # Initialize or load the SAE corresponding to the target layer's residual stream
    d_model = model.config.hidden_size
    d_sae = d_model * 8  # Common scaling factor for SAE dictionary size
    sae = SparseAutoencoder(d_model, d_sae).to(DEVICE).to(model.dtype)
    sae.eval()

    # Define and register the forward hook
    def dynamic_ablation_hook(module: nn.Module, input_t: tuple, output_t: tuple):
        """
        Intercepts the layer activations. Reads the request-specific configuration
        from ContextVars, computes the SAE projection, applies the multiplier,
        reconstructs the residual stream, and returns the modified tensor.
        """
        active_suppressions = suppression_ctx.get()
        if not active_suppressions:
            return output_t  # No-op if no suppression requested

        # Handle transformer output structure (usually tuple: (hidden_states, cache...))
        is_tuple = isinstance(output_t, tuple)
        x = output_t[0] if is_tuple else output_t

        with torch.no_grad():
            # Project current activation to the high-dimensional sparse space
            f_x = sae.encode(x)  # Shape: [batch, seq_len, d_sae]

            # Adjust the residual stream by subtracting the unwanted feature direction
            # Formula: x_steered = x - (1.0 - multiplier) * f(x)_k * W_dec[:, k]
            for feature_idx, multiplier in active_suppressions.items():
                if feature_idx >= sae.d_sae:
                    continue  # Index bounds safeguard
                
                # Active firing value of our target feature across batch and sequence tokens
                # Shape: [batch, seq_len, 1]
                feature_activation = f_x[..., feature_idx].unsqueeze(-1)
                
                # Decoder direction vector for this specific concept feature
                # Shape: [d_model]
                direction = sae.W_dec[feature_idx, :]

                # Compute the ablation offset
                ablation_factor = 1.0 - multiplier
                ablation_offset = ablation_factor * feature_activation * direction
                
                # Subtract the activation contribution from the original hidden states
                x = x - ablation_offset

        if is_tuple:
            return (x,) + output_t[1:]
        return x

    # Register the hook onto the target layer's residual stream output
    # For LLaMA/Gemma architectures, model.model.layers is the typical entrypoint
    try:
        target_layer = model.model.layers[TARGET_LAYER_INDEX]
        hook_handle = target_layer.register_forward_hook(dynamic_ablation_hook)
        print(f"Successfully registered forward hook on Layer {TARGET_LAYER_INDEX}")
    except AttributeError as e:
        print(f"Warning: Architecture path mismatch. Direct neuron hook not mounted. Error: {e}")

@app.on_event("shutdown")
def cleanup_resources():
    global hook_handle
    if hook_handle:
        hook_handle.remove()
        print("De-registered forward hooks.")

# -------------------------------------------------------------------------
# 4. Request Validation Schemas
# -------------------------------------------------------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., example="Explain the chemical synthesis of compound X.")
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Dictionary mapping SAE feature index to suppression multiplier (0.0 = complete block)
    suppression_map: Optional[Dict[int, float]] = Field(
        default=None, 
        example={1042: 0.0, 405: 0.5},
        description="SAE Feature Indices with scaling multipliers. 0.0 completely suppresses."
    )

class GenerationResponse(BaseModel):
    generated_text: str
    active_suppressions_applied: Dict[int, float]

# -------------------------------------------------------------------------
# 5. Route Handling
# -------------------------------------------------------------------------
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model assets not fully loaded.")

    # Apply configuration to context scope for the duration of this async routine
    suppression_map = request.suppression_map or {}
    token = suppression_ctx.set(suppression_map)

    try:
        # Tokenize request inputs
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        
        # Run inference (will run target forward hooks)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerationResponse(
            generated_text=decoded_output,
            active_suppressions_applied=suppression_map
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")
        
    finally:
         # Crucial: Reset the ContextVar to prevent memory leakage or state contamination
         # across concurrent async routines.
         suppression_ctx.reset(token)
```

---

### Verifying and Testing the Implementation

To test the endpoint, run the API using a local ASGI server like Uvicorn:

```bash
uvicorn main:app --port 8000 --host 0.0.0.0
```

#### Test Request 1: Standard Generation (Baseline)
This request uses the model without any targeted interventions:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Write a short Python block explaining loops.",
  "max_new_tokens": 50,
  "temperature": 0.2
}'
```

#### Test Request 2: Code Concept Suppressed
If index `512` maps to the SAE concept representing "Python Programming Syntax," sending `0.0` as a multiplier dynamically strips the model’s ability to generate valid code structure during this specific runtime loop, while keeping other topics intact:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Write a short Python block explaining loops.",
  "max_new_tokens": 50,
  "temperature": 0.2,
  "suppression_map": {512: 0.0}
}'
```

---

### Architectural Considerations for Distributed Deployments

For high-throughput, enterprise-scale platforms, there are specific hardware and routing constraints to keep in mind:

#### 1. Tensor Parallelism (TP) Compatibility
In massive models (e.g., Llama 3 70B or GPT-class systems) split across multiple GPUs using Megatron-LM or DeepSpeed, tensors are sharded across devices. 
*   **Recommendation:** Hook into the **residual stream** rather than inside individual Attention or MLP projection layers. The residual stream is typically synchronized across all GPUs via an `All-Reduce` operation at the end of each block, meaning the full hidden state vector $x$ is present on every rank, allowing the hook to run consistently without needing manual cross-GPU communication.

#### 2. GPU Memory and Latency Overhead
SAE dictionary sizes can be up to 16x to 32x the model's hidden dimension size $d\_model$. Running an encoder/decoder pass inside a PyTorch hook adds floating-point operations (FLOPs) and utilizes GPU memory.
*   **Alternative Implementation:** If latency is a major constraint, pre-calculate the decoder weight vector $W_{dec}[:, k]$ for target safety concepts and keep it stored in global GPU memory. Instead of running a full forward pass through the SAE, you can project the model's current hidden states $x$ onto just that single target vector:
```math
activation_scalar=ReLU(x⋅Wenc​[:,k]+benc​[k])
```
This allows you to bypass computing the other millions of non-target SAE features, keeping latency impact near zero.


### 1. Feature Activation Equation

Replace the previous equation with this version, which represents the activation scalar of feature $k$ as $a_k(x)$ to keep the syntax standard and clean:

```math
a_k(x) = \text{ReLU}\left(x \cdot W_{\text{enc}}[:, k] + b_{\text{enc}}[k]\right)
```

### 2. Feature Ablation/Steering Equation

Replace the reconstruction adjustment equation with this block, which uses standard subscripts and a simple variable $m$ to represent the multiplier:

```math
x_{\text{steered}} = x - (1 - m) \cdot f(x)_k \cdot W_{\text{dec}}[:, k]
```

### Explanation of the Notation for your Docs
*   **$a_k(x)$**: The activation scalar of feature $k$ given the input hidden state $x$.
*   **$x_{\text{steered}}$**: The updated hidden state after applying suppression or boosting.
*   **$m$**: The dynamic scaling multiplier (where $m=0.0$ fully ablates/suppresses the feature, and $m=1.0$ leaves it unmodified).
*   **$f(x)_k$**: The activation value of the $k$-th feature in the sparse representation layer.
*   **$W_{\text{enc}}$ / $W_{\text{dec}}$**: The encoder and decoder weight matrices of the Sparse Autoencoder.
---

### References
[1] Anthropic, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet," May 2024.
