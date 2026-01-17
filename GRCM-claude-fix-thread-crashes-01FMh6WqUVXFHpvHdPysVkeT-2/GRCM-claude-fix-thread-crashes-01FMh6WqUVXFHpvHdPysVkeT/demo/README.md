# 🌿 EchoZero Vine Memory - 3D Visualization Demo

**Real-time 3D visualization of EchoZero Spiral Lattice + Möbius Echo Layer dynamics**

A production-grade, browser-based demonstration showing:
- **Spiral Lattice**: Logarithmic spiral memory geometry
- **Möbius Echo Layer**: Topological consistency enforcement
- **Vine Growth**: Biological memory visualization
- **Torsion Fields**: Hallucination detection through geometry
- **Real-time Metrics**: Live coherence, torsion, and purity tracking

---

## 🎯 Quick Start

### **Option 1: Run with Real EchoZero Data** (Recommended)

```bash
# Terminal 1: Export real EchoZero state
cd /path/to/GRCM
python demo/export_echozero_state.py

# Terminal 2: Serve the demo
cd demo
python -m http.server 8000

# Browser: Open http://localhost:8000
```

### **Option 2: Run Demo Only** (No Python needed)

```bash
cd demo
python -m http.server 8000
# Open http://localhost:8000
```

The demo will run with simulated data if no `state.json` is found.

---

## 📁 Files

```
demo/
├── index.html                   # Main HTML page
├── main.js                      # 3D visualization engine
├── export_echozero_state.py     # Python state exporter
├── state.json                   # Real-time state (generated)
└── README.md                    # This file
```

---

## 🔧 Python State Exporter

### **Basic Usage**

```bash
# Export state at 30 FPS (default)
python export_echozero_state.py

# Export at 60 FPS
python export_echozero_state.py --fps 60

# Export for 60 seconds then stop
python export_echozero_state.py --duration 60

# Export single snapshot
python export_echozero_state.py --single
```

### **What It Does**

1. Runs **real EchoZero + Spiral Lattice + Möbius** dynamics
2. Exports state to `demo/state.json` every frame
3. Includes:
   - Spiral Lattice metrics (coherence, radius, angle, stabilizer)
   - Möbius Echo Layer metrics (torsion energy, damping gate, purity)
   - 3D position for visualization
   - Color-coded health status

### **Requirements**

**With PyTorch** (real implementation):
```bash
pip install torch numpy
```

**Without PyTorch** (theoretical simulation):
- Works with pure Python (no dependencies)
- Uses mathematical simulation instead of real tensors

---

## 🎨 3D Visualization Features

### **Visual Elements**

| Element | Description |
|---------|-------------|
| **Spiral Scaffold** | Semi-transparent reference spiral (logarithmic decay) |
| **Vine Nodes** | Growing memory points (color = coherence) |
| **Tendrils** | Smooth curves connecting recent memories |
| **Leaf Sprites** | High-coherence memory markers |
| **Bioelectric Pulses** | Torsion-triggered signals |
| **Grid** | Spatial reference plane |

### **Color Coding**

| Color | Meaning |
|-------|---------|
| 🟢 **Bright Green** | High coherence (>0.7) - healthy memory |
| 🟡 **Yellow-Green** | Moderate coherence (0.4-0.7) |
| 🟠 **Orange** | Low coherence (<0.4) - stressed |
| 🔴 **Red** | Critical torsion (hallucination) |

### **Controls**

| Input | Action |
|-------|--------|
| **Left Click + Drag** | Rotate camera |
| **Right Click + Drag** | Pan camera |
| **Scroll** | Zoom in/out |
| **Space** | Pause/Resume |

### **Buttons**

| Button | Function |
|--------|----------|
| **Inject Hallucination** | Triggers test hallucination (requires backend) |
| **Toggle Möbius** | Enable/disable Möbius layer (requires backend) |
| **Reset Camera** | Return to default view |
| **Record MP4** | Start/stop video recording |

---

## 📊 Metrics Panel

Displays real-time values from EchoZero:

| Metric | Description | Range |
|--------|-------------|-------|
| **Coherence** | Geometric alignment | 0.0 - 1.0 |
| **Torsion Energy** | Hallucination metric | 0.0 - 1.0 |
| **Spiral Radius** | Memory age indicator | 0.0 - 1.0 |
| **Spiral Angle** | Rotational position | 0 - 2π rad |
| **Möbius Gate** | Damping strength | 0.0 - 1.0 |
| **Memory Purity** | Consistency score | 0.0 - 1.0 |
| **Frame** | Current step number | 0 - ∞ |

---

## 📈 Graph Overlay

Bottom-left graph shows **coherence over time**:
- **X-axis**: Last 100 frames
- **Y-axis**: Coherence (0-1)
- **Green line**: Current trajectory
- **Green dot**: Current value

---

## 🎥 Recording Videos

1. Click **"🔴 Record MP4"**
2. Visualization continues normally
3. Click **"⏹️ Stop Recording"** when done
4. Browser downloads `.webm` video automatically

**Notes**:
- Recording captures at 30 FPS
- File format: WebM (modern browsers)
- Convert to MP4 with `ffmpeg` if needed:
  ```bash
  ffmpeg -i echozero_*.webm -c:v libx264 output.mp4
  ```

---

## 🔬 Technical Details

### **Architecture**

```
┌─────────────────────────────────────────────────────────┐
│ Python Backend (export_echozero_state.py)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Real EchoZero Implementation                    │   │
│  │  • Spiral Lattice (geometric memory)            │   │
│  │  • Möbius Echo Layer (topological consistency)  │   │
│  │  • Torsion computation                          │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Exports: state.json (30-60 FPS)                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Browser Frontend (index.html + main.js)                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Three.js 3D Visualization                       │   │
│  │  • Loads state.json every 33ms (~30 FPS)        │   │
│  │  • Renders vine growth animation                │   │
│  │  • Updates metrics panel                        │   │
│  │  • Draws coherence graph                        │   │
│  │  • Handles camera controls                      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### **Performance**

| Metric | Value |
|--------|-------|
| **Browser FPS** | 30-60 (depends on hardware) |
| **Python Export** | 30 FPS (configurable) |
| **Max Vine Nodes** | 800 (configurable in main.js) |
| **Memory Usage** | ~50 MB (browser) |
| **CPU Usage** | Low (5-15% on modern hardware) |

### **Compatibility**

| Platform | Status |
|----------|--------|
| **Chrome/Edge** | ✅ Fully supported |
| **Firefox** | ✅ Fully supported |
| **Safari** | ✅ Fully supported |
| **Mobile** | ⚠️ Limited (touch controls TBD) |

---

## 🎯 Use Cases

### **1. Technical Demonstrations**

Show how EchoZero's geometric memory works:
- **Spiral Lattice**: Temporal memory compression
- **Möbius Layer**: Hallucination detection
- **Torsion Energy**: Geometric truth metric

### **2. Presentations**

- **Investors**: Visual proof of concept
- **Engineers**: Architecture demonstration
- **Researchers**: Novel memory paradigm

### **3. Algorithm Debugging**

- Watch coherence in real-time
- Spot torsion spikes (hallucinations)
- Validate Möbius damping behavior
- Test parameter changes visually

### **4. Marketing Material**

- Record demo videos
- Create animated GIFs
- Show "living memory" concept
- Demonstrate biological computing aesthetic

---

## 🔧 Customization

### **Change Colors** (main.js)

```javascript
const COLORS = {
    background: 0x0a1f0f,   // Dark green-black
    healthy: 0x66ff88,      // Bright green
    stressed: 0xffcc55,     // Orange-yellow
    vine: 0x77ffaa,         // Teal-green
};
```

### **Adjust Camera** (main.js)

```javascript
const CONFIG = {
    cameraDistance: 55,     // Distance from origin
    autoRotate: true,       // Automatic rotation
    autoRotateSpeed: 0.2,   // Rotation speed
};
```

### **Change Vine Density** (main.js)

```javascript
const CONFIG = {
    vineMaxNodes: 800,      // Max visible nodes
    updateInterval: 33,     // Update rate (ms)
};
```

---

## 🐛 Troubleshooting

### **"State file not found" Error**

**Cause**: `state.json` doesn't exist yet

**Solution**:
```bash
# Run the Python exporter first
python export_echozero_state.py --single
```

### **Demo Runs But No Updates**

**Cause**: Python exporter not running

**Solution**:
```bash
# Start exporter in separate terminal
python export_echozero_state.py
```

### **Black Screen / No Visualization**

**Causes**:
1. Browser doesn't support WebGL
2. Graphics drivers outdated

**Solution**:
- Update browser to latest version
- Update graphics drivers
- Try different browser (Chrome recommended)

### **Recording Doesn't Work**

**Cause**: Browser security restrictions

**Solution**:
- Must serve from `http://localhost` (not `file://`)
- Use `python -m http.server` as shown above

---

## 📝 Development Notes

### **Adding New Features**

The demo is modular and extensible:

1. **Add new visuals**: Modify `main.js` → `addVineNode()` function
2. **Add new metrics**: Modify `export_echozero_state.py` → `export_state()` function
3. **Change layout**: Modify `index.html` CSS
4. **Add shaders**: Create new GLSL files (advanced)

### **Integration with Other Systems**

The demo can visualize any system that exports:

```json
{
  "step": 123,
  "position": {"x": 1.0, "y": 2.0, "z": 3.0},
  "coherence": 0.85,
  "torsion": 0.12,
  "spiral": { ... },
  "mobius": { ... }
}
```

Just modify `export_echozero_state.py` to export your data.

---

## 🚀 Next Steps

### **Enhancements**

- [ ] WebSocket for real-time streaming (no file I/O)
- [ ] Touch controls for mobile
- [ ] VR mode (WebXR)
- [ ] Shader-based torsion field visualization
- [ ] Multi-agent visualization (show multiple spirals)
- [ ] Export to glTF for 3D printing

### **Advanced Features**

- [ ] Particle system for pulses
- [ ] Bloom post-processing
- [ ] Audio synthesis (torsion → sound)
- [ ] Interactive hallucination injection UI
- [ ] Parameter sliders (real-time tuning)

---

## 📚 Related Documentation

- **Spiral Lattice Theory**: `/SPIRAL_LATTICE_INTEGRATION_GUIDE.md`
- **Möbius Echo Layer**: `/docs/MOBIUS_TOPOLOGY_THEORY.md`
- **EchoZero Overview**: `/docs/ECHOZERO_OVERVIEW.md`
- **Python Package**: `/grcm/echozero/`

---

## 🎬 Demo in Action

**Expected Visual Behavior**:

1. **Startup**: Spiral scaffold appears, camera slowly rotates
2. **First nodes**: Green vine nodes grow along the spiral
3. **Tendrils form**: Smooth curves connect recent memories
4. **Leaves appear**: High-coherence nodes sprout leaf sprites
5. **Pulses**: Occasional bioelectric pulses travel along vines
6. **Colors shift**: Based on coherence (green → yellow → orange)
7. **Graph updates**: Bottom-left shows coherence trajectory

**Healthy Operation**:
- Mostly green nodes
- Smooth tendril curves
- Low torsion (<0.1)
- Steady coherence (>0.7)

**Hallucination Event** (if injected):
- Orange/red nodes appear
- Torsion spikes (>0.5)
- Erratic pulse activity
- Coherence drops (<0.4)
- Möbius gate closes (low values)

---

## 💡 Tips

**For Best Results**:
1. Run Python exporter at 30 FPS (smooth but not overwhelming)
2. Use Chrome or Edge (best WebGL performance)
3. Full-screen the browser for immersive view
4. Let it run for 30-60 seconds to see full vine growth
5. Record a 10-20 second clip for presentations

**For Presentations**:
1. Start Python exporter first
2. Open demo in browser
3. Wait 10 seconds for initial growth
4. Click "Reset Camera" for clean view
5. Start recording
6. Let run for 15-20 seconds
7. Stop recording
8. Share the video!

---

## 📄 License

Part of the GRCM/EchoZero package.

---

## 🙏 Credits

**Visualization**: Three.js, OrbitControls
**Theory**: Spiral Lattice + Möbius Echo Layer architecture
**Aesthetic**: Biological "vine memory" concept

---

**Status**: ✅ Production Ready
**Version**: 1.0
**Last Updated**: November 30, 2025

---

*"From geometry to biology. From theory to vision."* 🌿
