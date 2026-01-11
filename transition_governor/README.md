# Transition Governor

**Deterministic control kernel for AI system stability**

## Overview

The Transition Governor is a standalone package that provides infrastructure-grade stability control for AI systems. It operates on a simple principle: **transition governance is a precondition, not a feature**.

This is not a model, agent, or symbolic system. This is infrastructure.

## Core Principles

1. **Transition governance is a precondition** - The governor exists independently and controls system behavior before semantic interpretation
2. **No interpretation inside the governor** - The governor operates only on measurable quantities: rates, magnitudes, authority, and limits
3. **No binary switching** - All transitions are continuous, rate-limited, and smoothly blended
4. **Graceful degradation over correctness** - Reduced capability is preferred to fabricated output

## Architecture

```
transition_governor/
├── core/                    # Control kernel
│   ├── state.py            # AIState, GovernanceState
│   ├── metrics.py          # Entropy & confidence computation
│   ├── governor.py         # TransitionGovernor (main kernel)
│   ├── authority.py        # Authority redistribution
│   └── degradation.py      # Brownout logic
├── adapters/               # LLM provider adapters
│   ├── hf_logits.py       # HuggingFace LogitsProcessor
│   └── openai_stream.py   # OpenAI streaming adapter
├── multi_agent/           # Multi-agent coordination
│   ├── agent.py           # Governed agent wrapper
│   ├── coordinator.py     # Multi-agent coordinator
│   └── allocator.py       # Global authority allocator
├── echozero_bridge/       # EchoZero integration
│   ├── contract.py        # Governed interface contract
│   └── runtime.py         # Execution wrapper
└── tests/                 # Test suite
    ├── test_governor.py
    ├── test_brownout.py
    ├── test_tool_failure.py
    └── test_multi_agent.py
```

## How It Works

### Input: AIState

The governor accepts a minimal state representation:

```python
@dataclass
class AIState:
    # Entropy signals
    entropy: float
    entropy_dot: float  # Rate of change

    # Confidence signals
    confidence: float
    confidence_dot: float

    # Tool state
    tool_state: ToolState

    # Resource usage
    context_length: int
    max_context_length: int

    # Fatigue
    fatigue: float
```

### Processing: Transition Intensity

The governor computes transition intensity as:

```
transition_intensity =
    w1 * |entropy_dot| +
    w2 * |confidence_dot| +
    w3 * |authority_shift|
```

### Decision: Stability Margin

```
remaining_margin = MAX_MARGIN - transition_intensity
```

If `remaining_margin < 0`, the system enters **brownout**.

### Output: Governed Constraints

```python
@dataclass
class GovernorOutput:
    governance_state: GovernanceState  # NORMAL | BROWNOUT
    authority_weights: Dict[str, float]
    confidence_cap: float
    max_tokens_per_step: Optional[int]
    max_tool_calls_per_step: int
```

## Brownout: Success State

**Brownout is not a failure mode**. It is a success state that prevents hallucination.

When in brownout:
- Confidence is capped aggressively (default: 0.3)
- Tool authority is zeroed (no external actions)
- Memory weight is reduced
- Verbosity limits enforced
- User is explicitly notified

## EchoZero Integration

The governor treats EchoZero as an **external governed client**.

### Contract

EchoZero MUST NEVER:
- Access raw logits
- Call tools directly
- Override governor outputs
- Escalate confidence independently

EchoZero MAY ONLY receive:
- `GovernedContext` (limited, capped, constrained)

### Enforcement

If EchoZero violates the contract, the runtime **hard-blocks** execution:

```python
runtime = GovernedEchoZeroRuntime(governor, echozero_client)

try:
    response, context = runtime.execute(state, user_input)
except ContractViolationError as e:
    # EchoZero violated governance constraints
    handle_violation(e)
```

## Multi-Agent Extension

The governor supports multi-agent systems with a simple rule:

```
authority(agent_i) ∝ stability(agent_i)
```

**No voting. No majority rule. No debate arbitration.**

Unstable agents lose influence smoothly.

```python
coordinator = MultiAgentCoordinator()
coordinator.add_agent(AgentConfig(agent_id="agent1"))
coordinator.add_agent(AgentConfig(agent_id="agent2"))

# Each round
results = coordinator.process_round({
    "agent1": state1,
    "agent2": state2
})

# Most stable agent dominates
dominant = coordinator.get_dominant_agent()
```

## Usage Examples

### Basic Usage

```python
from transition_governor import TransitionGovernor, AIState, ToolState

# Initialize governor
governor = TransitionGovernor(seed=42)

# Create state
state = AIState(
    entropy=1.5,
    entropy_dot=0.2,
    confidence=0.8,
    confidence_dot=-0.1,
    tool_state=ToolState.INACTIVE,
    context_length=500,
    max_context_length=2048,
    fatigue=0.0
)

# Govern
output = governor.govern(state)

print(f"State: {output.governance_state}")
print(f"Confidence cap: {output.confidence_cap}")
print(f"Authority: {output.authority_weights}")
```

### HuggingFace Integration

```python
from transformers import LogitsProcessorList
from transition_governor.adapters.hf_logits import GovernedLogitsProcessor

# Create governor
governor = TransitionGovernor()

# Create processor
processor = GovernedLogitsProcessor(governor)

# Use in generation
model.generate(
    input_ids=input_ids,
    logits_processor=LogitsProcessorList([processor]),
    max_length=100
)
```

### OpenAI Streaming

```python
from transition_governor.adapters.openai_stream import GovernedStreamWrapper

governor = TransitionGovernor()
wrapper = GovernedStreamWrapper(governor)

# Wrap stream
async for token in wrapper.wrap_stream(openai_stream, context_length=100):
    print(token, end='', flush=True)
```

### Multi-Agent Coordination

```python
from transition_governor.multi_agent import MultiAgentCoordinator, AgentConfig

coordinator = MultiAgentCoordinator()

# Add agents
coordinator.add_agent(AgentConfig(agent_id="reasoner"))
coordinator.add_agent(AgentConfig(agent_id="searcher"))
coordinator.add_agent(AgentConfig(agent_id="synthesizer"))

# Process round
results = coordinator.process_round({
    "reasoner": reasoner_state,
    "searcher": searcher_state,
    "synthesizer": synthesizer_state
})

# Get authority distribution
weights = coordinator.get_authority_distribution()
print(f"Authority: {weights}")

# Select response (from most stable agent)
selected = coordinator.select_response(agent_responses)
```

### Llama 3 Integration

For local inference with governed Llama 3 models:

```python
from transition_governor.examples.llama_integration import GovernedLlamaModel

# Initialize governed model
governed_model = GovernedLlamaModel(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda",
    governor_seed=42
)

# Generate with governance
text, governance_state, history = governed_model.generate_with_governance(
    prompt="Explain quantum computing in simple terms.",
    max_new_tokens=100
)

print(f"Generated: {text}")
print(f"Governance: {governance_state.value}")

# Analyze generation
analysis = governed_model.analyze_generation(history)
print(f"Mean entropy: {analysis['mean_entropy']:.2f}")
print(f"Brownout rate: {analysis['brownout_rate']*100:.1f}%")
```

**See [LLAMA_INTEGRATION.md](LLAMA_INTEGRATION.md) for complete integration guide including:**
- Hardware requirements
- Benchmark suite (TruthfulQA, MMLU, energy tests)
- Production deployment strategies
- Optimizations and troubleshooting

## Testing

All tests are deterministic and do not require interpretation:

```bash
# Run all tests
pytest transition_governor/tests/ -v

# Run specific test
pytest transition_governor/tests/test_brownout.py -v

# Run with coverage
pytest transition_governor/tests/ --cov=transition_governor
```

### Test Requirements

- **Deterministic replay**: Identical inputs → identical outputs
- **Tool failure → brownout**: No hallucination under tool errors
- **Long-context drift**: Confidence damping as context grows
- **Multi-agent containment**: Unstable agents lose authority

## Design Constraints

### What the Governor IS

- Infrastructure
- A control kernel
- Deterministic
- Auditable
- Boring

### What the Governor IS NOT

- A model
- An agent
- A symbolic reasoner
- A training loop
- An alignment system

## Failure Modes

The governor is designed to fail safely:

1. **Tool failure** → Enter brownout, block tools
2. **High entropy** → Cap confidence, reduce scope
3. **Context overflow** → Enforce brevity, limit memory
4. **Fatigue accumulation** → Gradual degradation
5. **Contract violation** → Hard block execution

## Deterministic Replay

All governor decisions can be replayed deterministically:

```python
# Capture history
states = [state1, state2, state3, ...]
outputs1 = [governor.govern(s) for s in states]

# Replay from history
outputs2 = governor.replay_from_history(states)

# Must be identical
assert outputs1 == outputs2
```

This is mandatory for audit and debugging.

## Dependencies

Minimal dependencies:
- `numpy` (for numerical computation)
- `typing` (for type hints)
- `dataclasses` (for state representation)

Optional:
- `transformers` (for HuggingFace adapter)
- `torch` (for HuggingFace adapter)
- `pytest` (for testing)

## License

[Specify license]

## Contributing

This is infrastructure. Contributions must prioritize:
1. Correctness
2. Determinism
3. Testability
4. Boring reliability

No features without corresponding tests.
No interpretation inside the governor.

## Citation

```bibtex
@software{transition_governor,
  title={Transition Governor: Deterministic Control Kernel for AI System Stability},
  author={[Author]},
  year={2025},
  url={https://github.com/[repo]/transition_governor}
}
```

## Acknowledgments

This work builds on principles from control theory, not AI alignment.
The governor enforces constraints, not values.
