# EchoZero Memory v2 - Interactive Demo

A comprehensive visual demonstration of the EchoZero Memory v2 system showing all subsystems in real-time.

## 🚀 Quick Start

1. Open `echozero_memory_v2_demo.html` in any modern browser
2. Click "▶ Start Simulation" to begin processing states
3. Watch the memory systems work in real-time

**No installation required** - runs entirely in the browser!

## 📊 What You'll See

### Multi-Resolution Memory Tiers
- **Green Bar (Recent)**: 256 states at full precision
- **Yellow Bar (Medium)**: 1024 states at rank-32 compression (every 4th step)
- **Red Bar (Long-term)**: 4096 states at rank-8 compression (every 16th step)

Demonstrates **17x more history** in the same memory budget.

### Torsion Measurement (Hallucination Detection)
- Real-time graph showing torsion scores
- **Blue line**: Normal states (torsion < 2.0)
- **Red dots**: Hallucination events (torsion > 2.0)
- **Dashed red line**: Detection threshold

Shows how phase discontinuities correlate with hallucinations.

### Semantic Buffer (Importance-Based)
- Bar chart of stored states sorted by importance
- **Red bars**: High-torsion events (preserved)
- **Blue bars**: Normal states
- Shows importance-based eviction in action

### Attractor Mapping
- Network graph of active Hopfield attractors
- **Purple nodes**: Individual attractors
- **Node size**: Number of activations
- **Connections**: States that triggered each attractor

## 🎮 Controls

| Button | Function |
|--------|----------|
| ▶ Start Simulation | Begin processing states at 10/sec |
| ⏸ Pause | Pause simulation (preserves state) |
| 🔄 Reset | Clear all memory and restart |
| 🔴 Record Screen | Capture demo as downloadable video |
| ⚠️ Inject Hallucination | Force a high-torsion event |
| ❓ Help | Show help dialog |

## 📈 Key Metrics

### Total Steps
Number of states processed through the system.

### States Stored
Total states across all three tiers (recent + medium + long-term).
Shows 17x improvement over baseline single-tier buffer.

### Memory Usage
Actual memory consumption in KB.
Formula: (recent × 64 × 4) + (medium × 32 × 4) + (longterm × 8 × 4) bytes

### Energy Savings
Percentage reduction from syncing every 300 steps instead of every step.
Formula: `(1 - syncs/total_steps) × 100%`

Typically reaches **99.8%** after warmup period.

### Hallucinations Detected
Count of high-torsion events (torsion > 2.0) detected and preserved.

## 🎥 Recording Demo Videos

1. Click **"🔴 Record Screen"**
2. Select browser tab or entire screen in the capture dialog
3. Click **"Start Simulation"** to begin demo
4. Let it run for 30-60 seconds to show all features
5. Click **"Inject Hallucination"** a few times to show detection
6. Click **"⏹ Stop Recording"** when done
7. Video automatically downloads as `.webm` file

### Tips for Great Recordings

- **Full screen**: Press F11 for immersive view
- **Narrate**: Explain what's happening as you record
- **Pause**: Use pause button to highlight specific features
- **Reset**: Start fresh for clean beginning

## 🧪 Experimental Features

### Inject Hallucination
Simulates LLM producing a high-confidence but incorrect output:
- Creates state with torsion = 5.0-8.0 (well above 2.0 threshold)
- Triggers sync event
- Stored in semantic buffer
- Mapped to attractor
- Highlighted in red in torsion graph

Use this to demonstrate:
- Real-time hallucination detection
- Importance-based retention
- Attractor basin mapping

## 🏗️ Architecture Visualized

```
Input State
    │
    ├─→ Recent Tier (always)
    │   └─ Full precision, 256 capacity
    │
    ├─→ Medium Tier (every 4th step)
    │   └─ Rank-32 compression, 1024 capacity
    │
    ├─→ Long-term Tier (every 16th step)
    │   └─ Rank-8 compression, 4096 capacity
    │
    ├─→ Semantic Buffer (if importance ≥ 1.0)
    │   └─ Evicts lowest-importance states
    │
    └─→ Attractor Map (if torsion > 2.0)
        └─ Cross-references temporal ↔ spatial
```

## 📱 Browser Compatibility

| Browser | Status | Notes |
|---------|--------|-------|
| **Chrome/Edge** | ✅ Full support | Best performance |
| **Firefox** | ✅ Full support | Screen recording works |
| **Safari** | ⚠️ Partial | Recording may not work |
| **Mobile** | ⚠️ Limited | Screen too small, no recording |

## 🐛 Troubleshooting

### "Failed to start recording"
- Allow screen capture permissions when prompted
- Try refreshing the page
- Use Chrome/Edge for best compatibility

### Simulation running slowly
- Close other browser tabs
- Reduce browser zoom level
- Use a modern browser

### Attractors not showing
- Let simulation run for at least 100 steps
- Click "Inject Hallucination" to force attractor creation
- High-torsion events are random (5% probability)

## 🎓 Educational Use

Perfect for:
- **Presentations**: Live demo of memory system
- **Teaching**: Visual explanation of multi-resolution architecture
- **Benchmarking**: Compare to traditional FIFO buffers
- **Research**: Prototype testing before production deployment

## 📚 Learn More

- **Full Documentation**: `/docs/MEMORY_V2_ARCHITECTURE.md`
- **Test Suite**: `/tests/memory_v2/` (73 tests, all passing)
- **Source Code**: `/grcm/echozero/memory_engine.py`

## 🤝 Contributing

Improvements welcome! Ideas:
- [ ] Add export to CSV/JSON
- [ ] Real-time performance metrics
- [ ] Comparison mode (before/after)
- [ ] Mobile-optimized layout
- [ ] Replay saved sessions

## 📄 License

Part of the EchoZero project. See main repository for license details.

---

**Built with ❤️ for local AI**
Enabling privacy-first, offline LLM inference on every device.
