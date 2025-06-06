# Plan: Per-Prompt Optimized Text-to-LoRA System

**I. System Goal & Core Principles**
    *   **Primary Goal:** Dynamically generate optimized LoRA weight matrices for a target diffusion model for any given text prompt.
    *   **Model-Agnosticism:** Adaptable via configuration (specifying target layers, dimensions, LoRA rank, and conditioning signals like text embedding dimensionality).
    *   **Efficiency:** Operates without involving the diffusion model's image latent space.
    *   **Modularity:** Distinct components for proposal, assembly, scoring, and optimization.
    *   **Per-Prompt Optimization:** LoRA generation involves an on-the-fly optimization loop tailored to each input prompt.

**II. Core System Modules**

    1.  **Text Input & Preprocessing Module:**
        *   Accepts raw text prompt.
        *   Tokenizes the prompt.

    2.  **Target Diffusion Model Configuration Module:**
        *   Loads, validates, and provides:
            *   `target_layers`: Specific layers in the diffusion model for LoRA.
            *   `layer_dimensions`: Original dimensions of weight matrices for target layers.
            *   `lora_rank` (`r`).
            *   `conditioning_signals`: E.g., target model's text embedding dimensionality, characteristics of its text encoder.
        *   Provides access to relevant components/computational graph segments of the target diffusion model (e.g., its text encoder, specific text-related attention layers) for the alignment scoring.

    3.  **LoRA Parameter Proposer (LLM-based):**
        *   Takes the tokenized prompt and `conditioning_signals`.
        *   Utilizes a pre-trained LLM to generate an *initial guess* or a parameterization of the LoRA matrices, providing a starting point for optimization.

    4.  **LoRA Matrix Assembler Module:**
        *   Takes raw parameters (from Proposer or Optimizer).
        *   Uses configuration (`target_layers`, `layer_dimensions`, `lora_rank`) to shape them into LoRA A (e.g., `dim_in x r`) and B (e.g., `r x dim_out`) matrices for each target layer.

    5.  **Alignment Scorer & Optimizer Module (Per-Prompt Training Engine):**
        *   **Function:** Iteratively refines LoRA parameters to maximize the alignment of the target model's (LoRA-modified) internal text representations with the input prompt.
        *   **Inputs:** Candidate LoRA matrices, original tokenized prompt, access to target model's text processing components.
        *   **Alignment Score Calculation (Conditional Language Modeling Objective):**
            *   Applies candidate LoRA to specified layers of the target model's text processing components.
            *   Processes the input prompt tokens through these LoRA-modified components.
            *   Derives the score from the **probability of the input prompt tokens given the (LoRA-modified) internal state/output of these components.** The optimization aims to maximize this probability (or minimize a corresponding loss like negative log-likelihood).
        *   **Optimization:** Uses the score/loss to update LoRA parameters (likely via gradient-based methods). Runs for a set number of iterations or until convergence for the current prompt.

    6.  **Output Module:**
        *   Packages the final, optimized LoRA matrices (e.g., in `.safetensors` format).

**III. High-Level Data Flow Diagram**

```mermaid
graph TD
    A[Text Prompt] --> B(Text Input & Preprocessing);
    C[Target Model Config File] --> D(Target Diffusion Model Configuration Module);

    B --> P(LoRA Parameter Proposer LLM);
    D -- Conditioning Signals --> P;

    P -- Initial/Parametrization --> ASM(LoRA Matrix Assembler Module);
    D -- Layer Specs --> ASM;

    subgraph "Per-Prompt Optimization Loop"
        direction LR
        ASM -- Candidate LoRA --> SC(Alignment Scorer);
        B -- Original Prompt --> SC;
        D -- Target Model Access/Segments --> SC;
        SC -- Alignment Score/Loss --> OPT(Optimizer);
        OPT -- Updated Params --> ASM;
    end

    OPT -- Final Optimized LoRA --> OUT(Output Module);
    OUT --> G[Generated LoRA Weights];