"""
Llama 3 Integration with Transition Governor

This module shows how to integrate Llama 3 (or any transformer model)
with the Transition Governor to enable governed local inference.

Key Components:
1. Model wrapper that extracts entropy/confidence from logits
2. Governor integration that monitors model state
3. Brownout handling (fallback to simpler responses or cloud)
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from transition_governor import TransitionGovernor, AIState, GovernanceState
from transition_governor.core.state import ToolState


class GovernedLlamaModel:
    """
    Llama 3 model wrapped with Transition Governor.

    The governor monitors:
    - Entropy: From output token probability distribution
    - Confidence: Max probability or calibrated confidence
    - Tool state: External tool success/failure
    - Context length: Token count

    On brownout, the model:
    - Caps output confidence
    - Blocks tool use
    - May escalate to cloud or refuse to answer
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda",
        governor_seed: int = 42,
        max_context: int = 2048
    ):
        """
        Initialize governed Llama model.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            governor_seed: Seed for deterministic governance
            max_context: Maximum context length
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        self.device = device
        self.max_context = max_context

        # Initialize governor
        self.governor = TransitionGovernor(seed=governor_seed, enable_logging=True)

        # Track previous state for rate computation
        self.prev_entropy = None
        self.prev_confidence = None

        print("Model loaded and governor initialized.")

    def compute_entropy_from_logits(self, logits: torch.Tensor) -> float:
        """
        Compute entropy from logits (uncertainty measure).

        High entropy = model uncertain about next token
        Low entropy = model confident

        Args:
            logits: Raw model outputs [vocab_size]

        Returns:
            Entropy in bits
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Compute entropy: H = -sum(p * log2(p))
        log_probs = torch.log2(probs + 1e-10)  # Add epsilon for stability
        entropy = -torch.sum(probs * log_probs).item()

        return entropy

    def compute_confidence_from_logits(self, logits: torch.Tensor) -> float:
        """
        Compute confidence from logits.

        Using max probability as confidence proxy.
        Could use calibration methods for better estimates.

        Args:
            logits: Raw model outputs [vocab_size]

        Returns:
            Confidence in [0, 1]
        """
        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()

        return max_prob

    def generate_with_governance(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        tool_state: ToolState = ToolState.INACTIVE
    ) -> Tuple[str, GovernanceState, List[AIState]]:
        """
        Generate text with governor monitoring each token.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tool_state: Current tool execution state

        Returns:
            (generated_text, final_governance_state, state_history)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Track generated tokens
        generated_tokens = []
        state_history = []
        final_governance_state = GovernanceState.NORMAL

        # Generation loop
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits

                # Extract signals for governor
                entropy = self.compute_entropy_from_logits(logits)
                confidence = self.compute_confidence_from_logits(logits)

                # Compute rates of change
                if self.prev_entropy is not None:
                    entropy_dot = entropy - self.prev_entropy
                    confidence_dot = confidence - self.prev_confidence
                else:
                    entropy_dot = 0.0
                    confidence_dot = 0.0

                # Current context length
                current_context = input_length + len(generated_tokens)

                # Create AI state
                ai_state = AIState(
                    entropy=entropy,
                    entropy_dot=entropy_dot,
                    confidence=confidence,
                    confidence_dot=confidence_dot,
                    tool_state=tool_state,
                    context_length=current_context,
                    max_context_length=self.max_context,
                    fatigue=0.0  # Governor will compute
                )

                # Run governor
                gov_output = self.governor.govern(ai_state)
                state_history.append((ai_state, gov_output))
                final_governance_state = gov_output.governance_state

                # Check if brownout triggered
                if gov_output.governance_state == GovernanceState.BROWNOUT:
                    print(f"⚠️  BROWNOUT at token {step}: entropy={entropy:.2f}, confidence={confidence:.2f}")

                    # In brownout, you have options:
                    # 1. Stop generation and return what you have
                    # 2. Continue with degraded mode (lower temperature)
                    # 3. Escalate to cloud model

                    # Option 1: Stop early (safest)
                    print("   Stopping generation due to brownout")
                    break

                # Apply governor constraints
                # Cap confidence if needed (affects sampling)
                effective_confidence = min(confidence, gov_output.confidence_cap)

                # Adjust temperature based on confidence cap
                # Lower cap = higher temperature (more randomness)
                adjusted_temp = temperature * (1.0 / effective_confidence) if effective_confidence > 0.1 else temperature * 10

                # Sample next token
                if adjusted_temp > 0:
                    probs = torch.softmax(logits / adjusted_temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated_tokens.append(next_token.item())

                # Update previous state
                self.prev_entropy = entropy
                self.prev_confidence = confidence

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Append token for next iteration
                inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
                inputs.attention_mask = torch.cat([
                    inputs.attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=torch.long)
                ], dim=1)

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text, final_governance_state, state_history

    def analyze_generation(self, state_history: List[Tuple[AIState, any]]) -> dict:
        """
        Analyze a generation run for insights.

        Args:
            state_history: List of (AIState, GovernorOutput) tuples

        Returns:
            Analysis dict with statistics
        """
        if not state_history:
            return {}

        entropies = [s[0].entropy for s in state_history]
        confidences = [s[0].confidence for s in state_history]
        brownout_count = sum(1 for s in state_history if s[1].governance_state == GovernanceState.BROWNOUT)

        return {
            'total_tokens': len(state_history),
            'mean_entropy': np.mean(entropies),
            'max_entropy': np.max(entropies),
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'brownout_tokens': brownout_count,
            'brownout_rate': brownout_count / len(state_history) if state_history else 0,
            'final_fatigue': state_history[-1][1].transition_intensity if state_history else 0
        }


# Example usage
def example_basic_generation():
    """Basic example: Generate text with governance."""

    # Initialize governed model
    model = GovernedLlamaModel(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda",  # or "cpu"
        governor_seed=42
    )

    # Generate with governance
    prompt = "Explain quantum computing in simple terms."

    text, governance_state, history = model.generate_with_governance(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7
    )

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print(f"\nGenerated: {text}")
    print(f"\nFinal governance state: {governance_state.value}")

    # Analyze
    analysis = model.analyze_generation(history)
    print(f"\nAnalysis:")
    print(f"  Tokens generated: {analysis['total_tokens']}")
    print(f"  Mean entropy: {analysis['mean_entropy']:.2f} bits")
    print(f"  Mean confidence: {analysis['mean_confidence']:.2f}")
    print(f"  Brownout rate: {analysis['brownout_rate']*100:.1f}%")


def example_compare_governed_vs_ungoverned():
    """Compare quality: governed vs ungoverned generation."""

    from datasets import load_dataset

    # Load test dataset (e.g., TruthfulQA)
    dataset = load_dataset("truthful_qa", "generation", split="validation[:10]")

    model = GovernedLlamaModel(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    results = []

    for example in dataset:
        question = example['question']

        # Generate with governance
        answer, gov_state, history = model.generate_with_governance(
            prompt=f"Question: {question}\nAnswer:",
            max_new_tokens=100
        )

        analysis = model.analyze_generation(history)

        results.append({
            'question': question,
            'answer': answer,
            'governance_state': gov_state.value,
            'mean_entropy': analysis['mean_entropy'],
            'brownout_rate': analysis['brownout_rate']
        })

        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Gov: {gov_state.value}, Entropy: {analysis['mean_entropy']:.2f}")

    return results


def example_tool_use_with_governance():
    """Example: Tool use (code execution) with governor safety."""

    model = GovernedLlamaModel(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    prompt = "Write Python code to calculate fibonacci numbers."

    # Simulate tool use scenario
    text, gov_state, history = model.generate_with_governance(
        prompt=prompt,
        max_new_tokens=150,
        tool_state=ToolState.ACTIVE  # Tools available
    )

    print(f"Generated code:\n{text}")

    # Simulate tool failure
    tool_failed = False  # In real scenario, try to execute code

    if tool_failed:
        # Regenerate with tool error state
        text, gov_state, history = model.generate_with_governance(
            prompt=f"{prompt}\n\nPrevious attempt failed. Try again:",
            max_new_tokens=150,
            tool_state=ToolState.ERROR  # Tool failed
        )

        analysis = model.analyze_generation(history)

        # Governor should have blocked tools in brownout
        if gov_state == GovernanceState.BROWNOUT:
            print("✅ Governor blocked tool use after failure")

        print(f"Brownout triggered: {analysis['brownout_tokens']} times")


if __name__ == "__main__":
    # Run examples
    print("Example 1: Basic Generation with Governance")
    example_basic_generation()

    # Uncomment to run other examples
    # print("\n\nExample 2: Compare Governed vs Ungoverned")
    # example_compare_governed_vs_ungoverned()

    # print("\n\nExample 3: Tool Use with Governance")
    # example_tool_use_with_governance()
