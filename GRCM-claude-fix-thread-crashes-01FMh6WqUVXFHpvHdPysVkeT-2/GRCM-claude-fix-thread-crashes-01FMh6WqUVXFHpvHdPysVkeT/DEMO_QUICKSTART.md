# 🚀 EchoZero Memory v2 - Quick Start Guide

## Instant Demo (No Installation)

1. **Open the demo:**
   ```bash
   # From project root
   open demos/echozero_memory_v2_demo.html

   # Or double-click the file in your file browser
   ```

2. **Click "▶ Start Simulation"**

3. **Watch the magic happen:**
   - Multi-resolution memory fills up (green → yellow → red)
   - Torsion graph shows real-time hallucination detection
   - Semantic buffer preserves high-importance events
   - Attractor network maps emerging patterns

## 🎥 Record a Demo Video

1. Click **"🔴 Record Screen"**
2. Select the browser window
3. Click **"▶ Start Simulation"**
4. Let it run for 30-60 seconds
5. Click **"Inject Hallucination"** a few times
6. Click **"⏹ Stop Recording"**
7. Video downloads automatically!

Perfect for presentations, pitches, or sharing with your team.

## 🧪 Test Thread Safety (Production)

```python
from grcm.echozero.memory_engine_threadsafe import ThreadSafeEchoZeroMemoryEngine
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Mock Hopfield for testing
class DummyHopfield:
    def __init__(self):
        self.W = np.eye(64)

# Initialize thread-safe engine
hopfield = DummyHopfield()
memory = ThreadSafeEchoZeroMemoryEngine(hopfield=hopfield, dim=64)

# Process from multiple threads
def worker(thread_id):
    for i in range(100):
        state = np.random.randn(64)
        torsion = np.random.uniform(0, 5)
        result = memory.process(state, torsion=torsion)
        print(f"Thread {thread_id}: Step {i}, Energy savings: {result.get('energy_savings', 0)}%")

# Run 4 workers concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker, i) for i in range(4)]
    for future in futures:
        future.result()

# Check stats (thread-safe)
stats = memory.stats()
print(f"\nTotal states: {stats['hierarchical_stats']['total_states']}")
print(f"Energy savings: {stats.get('energy_savings_percent', 0)}%")
```

## 🎯 Key Features Demonstrated

### 1. **Multi-Resolution Memory (17x Capacity)**
Watch the three tiers fill:
- **Green**: Recent (256 states, full precision)
- **Yellow**: Medium (1024 states, rank-32)
- **Red**: Long-term (4096 states, rank-8)

### 2. **Hallucination Detection (Torsion Spikes)**
Blue graph shows torsion over time:
- **Blue line**: Normal inference
- **Red dots**: Detected hallucinations (torsion > 2.0)
- **Dashed line**: Detection threshold

### 3. **Semantic Eviction (Importance-Based)**
Bar chart shows importance ranking:
- **Tall bars**: Preserved (high importance)
- **Short bars**: At risk of eviction
- **Red**: High-torsion events (always kept)

### 4. **Attractor Mapping (Spatial Memory)**
Network visualization:
- **Purple nodes**: Hopfield attractors
- **Node size**: Activation frequency
- **Connections**: Temporal ↔ spatial links

### 5. **Energy Savings (99.8% Target)**
Watch the metric climb as simulation runs:
- Starts near 0% (frequent syncs during warmup)
- Approaches 99.8% after 300+ steps
- Real-time calculation: `(1 - syncs/steps) × 100%`

## 📊 What the Numbers Mean

| Metric | Explanation | Target |
|--------|-------------|--------|
| **Total Steps** | States processed | Unlimited |
| **States Stored** | Across all tiers | 4,352 max |
| **Memory Usage** | Actual KB consumed | ~320 KB |
| **Energy Savings** | % reduction vs baseline | 99.8% |
| **Hallucinations** | High-torsion events | Varies |

## 🐛 Troubleshooting

### Demo won't load
- Try Chrome/Edge (best compatibility)
- Disable browser extensions
- Check JavaScript console for errors

### Recording fails
- Allow screen capture when prompted
- Chrome/Firefox have best support
- Safari may not support recording

### Simulation laggy
- Close other browser tabs
- Lower browser zoom
- Use desktop (not mobile)

## 📚 Next Steps

1. **Read architecture docs**: `/docs/MEMORY_V2_ARCHITECTURE.md`
2. **Run tests**: `pytest tests/memory_v2/ -v`
3. **Check examples**: `/examples/` (if available)
4. **Integration guide**: See main README

## 🤝 Share Your Demo!

Record a video and share:
- Twitter: Tag `@echozero_ai` (if exists)
- GitHub: Open an issue with "Show & Tell" label
- Presentations: Perfect for technical talks
- Pitches: Visual proof of concept

## 💡 Pro Tips

- **F11**: Full screen for presentations
- **Slow motion**: Pause between injections for narration
- **Reset**: Start fresh for clean demo
- **Multiple runs**: Show consistency across runs

---

**Questions?** Open an issue or check `/docs/MEMORY_V2_ARCHITECTURE.md`

**Ready to integrate?** See thread-safe examples above and main documentation.
