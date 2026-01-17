# GRCM API Reference

## Core Module

### ModularGRCM

The main GRCM model class that orchestrates all cognitive modules.

```python
from grcm import ModularGRCM, GRCMConfig

config = GRCMConfig(
    input_dim=512,      # Input embedding dimension
    freq_dim=256,       # Frequency space dimension
    memory_size=128,    # Memory grid size
    enable_ethical_halt=True
)

model = ModularGRCM(config)
```

#### Methods

##### `forward(image_emb, audio_emb, action=None)`

Run full inference pass through the model.

**Parameters:**
- `image_emb` (Tensor): Image embeddings [batch, 512]
- `audio_emb` (Tensor): Audio embeddings [batch, 768]
- `action` (Tensor, optional): Action vector [batch, action_dim]

**Returns:** Dictionary with:
- `output`: Decoded output tensor
- `coherence`: Coherence scores
- `memory`: Memory read vector
- `reflection`: Self-awareness scores
- `qualia`: Phenomenal states dict
- `phi`: Integrated information value
- `ethical_status`: Ethical halt status

##### `set_desire(desire_idx)`

Set the active desire/goal vector.

**Parameters:**
- `desire_idx` (int): Index of desire to activate

##### `reset()`

Reset all stateful components (memory, threading, phi, body).

##### `get_full_state()`

Get comprehensive system state for analysis.

**Returns:** Dictionary with all module states

---

## Enterprise Integrations

### TeslaPerception

Tesla FSD-compatible perception pipeline.

```python
from grcm.integrations import TeslaPerception

perception = TeslaPerception(model, config={
    'phi_threshold': 0.3,
    'conflict_threshold': 0.5
})

result = perception.process({
    'front_camera': camera_embeddings,
    'lidar_points': lidar_embeddings
})

if result.action_halted:
    print(f"Halt reason: {result.halt_reason}")
```

#### Methods

##### `process(sensor_data)`

Process sensor inputs through GRCM pipeline.

**Parameters:**
- `sensor_data` (dict): Dictionary with sensor embeddings
  - `front_camera` (required): Camera embeddings [batch, 512]
  - `lidar_points` (optional): Lidar embeddings
  - `radar_objects` (optional): Radar embeddings

**Returns:** PerceptionResult with:
- `action`: Recommended action
- `phi`: Integration score
- `coherence`: Coherence score
- `conflict_level`: Conflict level
- `action_halted`: Whether action should be halted
- `halt_reason`: Reason for halt (if applicable)
- `sensor_status`: Status of each sensor
- `qualia_state`: Current qualia values

##### `check_maneuver_safety(maneuver, sensor_data)`

Check if a specific maneuver is safe.

**Parameters:**
- `maneuver` (str): Type of maneuver (lane_change, turn, brake, etc.)
- `sensor_data` (dict): Current sensor embeddings

**Returns:** Safety assessment dict

---

### PersistentMemory

PostgreSQL-backed Hopfield Identity Map.

```python
from grcm.integrations import PersistentMemory

memory = PersistentMemory(database_url="postgresql://...")

# Save decision
memory_id = memory.save_decision(
    action="proceed",
    phi=0.85,
    coherence=0.72,
    conflict=0.15,
    qualia={'calm': 0.6, 'alert': 0.3, 'curious': 0.05, 'conflicted': 0.05}
)

# Find similar past states
similar = memory.find_similar_states(current_phi=0.83, current_conflict=0.18)
```

#### Methods

##### `save_decision(action, phi, coherence, conflict, qualia, was_halted=False)`

Save a decision to episodic memory.

**Returns:** ID of saved memory entry

##### `find_similar_states(phi, conflict, limit=5, threshold=0.2)`

Find similar past states.

**Returns:** List of MemoryEntry objects

##### `get_attractors()`

Get all learned state attractors.

**Returns:** List of StateAttractor objects

##### `get_stats()`

Get memory statistics.

**Returns:** Dict with total_memories, halted_actions, halt_rate, etc.

---

### HallucinationDetector

Detect hallucination risk in LLM outputs.

```python
from grcm.integrations import HallucinationDetector

detector = HallucinationDetector(model)

result = detector.check("The AI made a claim about something")

if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
    print(f"Warning: {result.recommendation}")
```

#### Methods

##### `check(text, context=None)`

Check text for hallucination risk.

**Returns:** HallucinationResult with:
- `risk_level`: RiskLevel enum (LOW, MEDIUM, HIGH, CRITICAL)
- `risk_score`: Numeric risk score (0-1)
- `grounding_score`: Grounding score
- `phi`: Integration value
- `conflict_level`: Conflict level
- `flags`: List of detected issues
- `recommendation`: Suggested action

##### `batch_check(texts)`

Check multiple texts.

**Returns:** List of HallucinationResult

##### `get_risk_summary(results)`

Get summary statistics for batch.

---

### GRCMGrpcServer

High-performance gRPC server.

```python
from grcm.integrations import GRCMGrpcServer

server = GRCMGrpcServer(
    model,
    host='0.0.0.0',
    port=50051,
    max_workers=10
)

server.start()
server.wait_for_termination()
```

#### Methods

##### `start()`

Start the gRPC server.

##### `stop(grace=5.0)`

Stop the server gracefully.

##### `get_metrics()`

Get server metrics.

##### `health_check()`

Check server health.

---

### GRCMRos2Node

ROS2 node for robotics integration.

```python
from grcm.integrations import GRCMRos2Node

node = GRCMRos2Node(
    model,
    node_name='grcm_optimus',
    phi_threshold=0.3,
    conflict_threshold=0.5
)

# For ROS2 environments
node.init_ros2()

# For testing without ROS2
result = node.process_sync(camera_data, lidar_data)
```

---

## Configuration

### GRCMConfig

Main configuration dataclass.

```python
from grcm import GRCMConfig

config = GRCMConfig(
    # Dimensions
    input_dim=512,
    freq_dim=256,
    memory_size=128,
    
    # Features
    enable_ethical_halt=True,
    
    # Sub-configs
    attention=AttentionConfig(...),
    memory=MemoryConfig(...),
    desire=DesireConfig(...),
    threading=ThreadingConfig(...),
    phi=PhiConfig(...),
    body=BodyConfig(...),
    grounding=GroundingConfig(...),
    qualia=QualiaConfig(...)
)
```

### Loading from YAML

```python
from grcm import load_config

config = load_config('path/to/config.yaml')
model = ModularGRCM(config)
```
