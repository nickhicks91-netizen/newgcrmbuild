/**
 * EchoZero Vine Memory - 3D Visualization
 * Main Application Entry Point
 */

// Configuration
const CONFIG = {
    stateFile: 'state.json',
    updateInterval: 33, // ~30 FPS
    vineMaxNodes: 800,
    cameraDistance: 55,
    autoRotate: true,
    autoRotateSpeed: 0.2,
};

// Global state
let scene, camera, renderer, controls;
let vineNodes = [];
let tendrilLines = [];
let pulses = [];
let currentState = null;
let paused = false;
let recording = false;
let mediaRecorder = null;
let recordedChunks = [];

// Colors (biological/organic palette)
const COLORS = {
    background: 0x0a1f0f,
    healthy: 0x66ff88,      // Bright green
    moderate: 0xaaff77,     // Yellow-green
    stressed: 0xffcc55,     // Orange-yellow
    critical: 0xff8855,     // Orange-red
    vine: 0x77ffaa,         // Soft teal-green
    pulse: 0x99ffdd,        // Cyan glow
    lattice: 0x88ff99,      // Lattice points
};

// Graph for coherence tracking
let graphCanvas, graphCtx, graphData = [];

/**
 * Initialize Three.js scene
 */
function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);
    scene.fog = new THREE.FogExp2(COLORS.background, 0.015);

    // Camera
    camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        2000
    );
    camera.position.set(0, 20, CONFIG.cameraDistance);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = CONFIG.autoRotate;
    controls.autoRotateSpeed = CONFIG.autoRotateSpeed;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    const backLight = new THREE.DirectionalLight(0x66ff88, 0.3);
    backLight.position.set(-10, -10, -10);
    scene.add(backLight);

    // Grid helper (subtle)
    const gridHelper = new THREE.GridHelper(100, 20, 0x1e6032, 0x1e6032);
    gridHelper.material.opacity = 0.15;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    // Spiral lattice scaffold (static reference)
    createSpiralScaffold();

    // Window resize
    window.addEventListener('resize', onWindowResize);

    console.log('✅ Scene initialized');
}

/**
 * Create static spiral lattice scaffold
 */
function createSpiralScaffold() {
    const N = 256;
    const points = [];

    for (let i = 0; i < N; i++) {
        const t = i;
        const r = Math.exp(-0.015 * t);
        const theta = 0.45 * t;
        const y = (t / N) * 20 - 10;

        const x = r * Math.cos(theta) * 10;
        const z = r * Math.sin(theta) * 10;

        points.push(new THREE.Vector3(x, y, z));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: COLORS.lattice,
        opacity: 0.2,
        transparent: true,
    });

    const line = new THREE.Line(geometry, material);
    scene.add(line);
}

/**
 * Initialize graph overlay
 */
function initGraph() {
    graphCanvas = document.getElementById('graph');
    graphCtx = graphCanvas.getContext('2d');
}

/**
 * Update graph with new coherence value
 */
function updateGraph(coherence) {
    graphData.push(coherence);
    if (graphData.length > 100) {
        graphData.shift();
    }

    const ctx = graphCtx;
    const w = graphCanvas.width;
    const h = graphCanvas.height;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Draw grid
    ctx.strokeStyle = 'rgba(119, 255, 170, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Draw coherence line
    ctx.strokeStyle = '#88ffaa';
    ctx.lineWidth = 2;
    ctx.beginPath();

    graphData.forEach((val, i) => {
        const x = (i / 100) * w;
        const y = h - (val * h);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw current value indicator
    if (graphData.length > 0) {
        const lastVal = graphData[graphData.length - 1];
        const x = w - 2;
        const y = h - (lastVal * h);

        ctx.fillStyle = '#66ff88';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

/**
 * Load state from JSON file
 */
async function loadState() {
    try {
        const response = await fetch(CONFIG.stateFile + '?t=' + Date.now());
        if (!response.ok) {
            throw new Error('State file not found');
        }
        const state = await response.json();
        return state;
    } catch (error) {
        console.warn('Could not load state:', error.message);
        return null;
    }
}

/**
 * Update UI metrics
 */
function updateUI(state) {
    if (!state) return;

    document.getElementById('coherence').textContent = state.coherence.toFixed(3);
    document.getElementById('torsion').textContent = state.torsion.toFixed(4);
    document.getElementById('radius').textContent = state.spiral.radius.toFixed(4);
    document.getElementById('angle').textContent = (state.spiral.angle % (2 * Math.PI)).toFixed(3);
    document.getElementById('gate').textContent = state.mobius.gate_mean.toFixed(3);
    document.getElementById('purity').textContent = state.mobius.purity.toFixed(3);
    document.getElementById('frame').textContent = state.step;

    // Color-code coherence
    const cohEl = document.getElementById('coherence');
    cohEl.classList.remove('high', 'low', 'critical');
    if (state.coherence > 0.7) {
        cohEl.classList.add('high');
    } else if (state.coherence < 0.4) {
        cohEl.classList.add('critical');
    } else {
        cohEl.classList.add('low');
    }

    // Update status
    document.getElementById('status-text').textContent =
        `Live • ${state.mode === 'pytorch' ? 'PyTorch' : 'Theoretical'}`;
}

/**
 * Add new vine node to scene
 */
function addVineNode(state) {
    const position = state.position;
    const color = getColorForState(state);
    const size = state.coherence > 0.7 ? 0.35 : 0.2;

    // Create node geometry
    const geometry = new THREE.SphereGeometry(size, 12, 12);
    const material = new THREE.MeshPhongMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.9,
    });

    const node = new THREE.Mesh(geometry, material);
    node.position.set(position.x, position.y, position.z);

    // Add glow effect for high coherence
    if (state.coherence > 0.8) {
        const glowGeometry = new THREE.SphereGeometry(size * 1.5, 8, 8);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: COLORS.healthy,
            transparent: true,
            opacity: 0.2,
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        node.add(glow);
    }

    scene.add(node);
    vineNodes.push(node);

    // Limit total nodes
    if (vineNodes.length > CONFIG.vineMaxNodes) {
        const oldNode = vineNodes.shift();
        scene.remove(oldNode);
    }

    // Maybe create leaf sprite
    if (Math.random() < 0.12 && state.coherence > 0.6) {
        createLeafSprite(position);
    }

    // Update tendril curve
    updateTendril(position);
}

/**
 * Get color based on state
 */
function getColorForState(state) {
    if (state.color === 'healthy') return COLORS.healthy;
    if (state.color === 'moderate') return COLORS.moderate;
    if (state.color === 'stressed') return COLORS.stressed;
    return COLORS.critical;
}

/**
 * Create leaf sprite
 */
function createLeafSprite(position) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Draw simple leaf shape
    ctx.fillStyle = '#9affb2';
    ctx.beginPath();
    ctx.ellipse(32, 32, 24, 16, Math.PI / 4, 0, Math.PI * 2);
    ctx.fill();

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0.7,
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.8, 0.8, 0.8);
    sprite.position.set(
        position.x + (Math.random() - 0.5) * 0.5,
        position.y + (Math.random() - 0.5) * 0.5,
        position.z + (Math.random() - 0.5) * 0.5
    );

    scene.add(sprite);
    vineNodes.push(sprite);
}

/**
 * Update tendril curve (vine connections)
 */
function updateTendril(newPosition) {
    const recentNodes = vineNodes.slice(-20);
    if (recentNodes.length < 2) return;

    // Remove old tendril
    if (tendrilLines.length > 0) {
        const oldLine = tendrilLines.shift();
        scene.remove(oldLine);
    }

    // Create curve from recent nodes
    const points = recentNodes.map(node =>
        new THREE.Vector3(node.position.x, node.position.y, node.position.z)
    );

    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.08, 8, false);
    const tubeMaterial = new THREE.MeshPhongMaterial({
        color: COLORS.vine,
        transparent: true,
        opacity: 0.6,
        emissive: COLORS.vine,
        emissiveIntensity: 0.1,
    });

    const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
    scene.add(tube);
    tendrilLines.push(tube);

    // Limit tendrils
    if (tendrilLines.length > 5) {
        const old = tendrilLines.shift();
        scene.remove(old);
    }
}

/**
 * Create bioelectric pulse
 */
function createPulse(start, end) {
    const geometry = new THREE.SphereGeometry(0.15, 8, 8);
    const material = new THREE.MeshBasicMaterial({
        color: COLORS.pulse,
        transparent: true,
        opacity: 0.9,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(start);

    scene.add(mesh);

    pulses.push({
        mesh: mesh,
        start: start.clone(),
        end: end.clone(),
        t: 0,
    });
}

/**
 * Update all pulses
 */
function updatePulses() {
    pulses = pulses.filter(pulse => {
        pulse.t += 0.03;

        if (pulse.t >= 1.0) {
            scene.remove(pulse.mesh);
            return false;
        }

        pulse.mesh.position.lerpVectors(pulse.start, pulse.end, pulse.t);
        pulse.mesh.material.opacity = 0.9 * (1.0 - pulse.t);

        return true;
    });
}

/**
 * Animation loop
 */
function animate() {
    requestAnimationFrame(animate);

    if (!paused) {
        controls.update();
        updatePulses();
    }

    renderer.render(scene, camera);
}

/**
 * State update loop
 */
async function stateLoop() {
    if (!paused) {
        const state = await loadState();

        if (state) {
            currentState = state;
            updateUI(state);
            addVineNode(state);
            updateGraph(state.coherence);

            // Random pulses based on torsion
            if (Math.random() < state.torsion * 0.2 && vineNodes.length > 10) {
                const start = vineNodes[vineNodes.length - 1].position.clone();
                const end = vineNodes[Math.max(0, vineNodes.length - 10)].position.clone();
                createPulse(start, end);
            }
        }
    }

    setTimeout(stateLoop, CONFIG.updateInterval);
}

/**
 * Window resize handler
 */
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

/**
 * Setup UI event listeners
 */
function setupUI() {
    // Record button
    document.getElementById('record').addEventListener('click', toggleRecording);

    // Reset camera
    document.getElementById('reset-view').addEventListener('click', () => {
        camera.position.set(0, 20, CONFIG.cameraDistance);
        controls.reset();
    });

    // Toggle Möbius (placeholder - would need backend integration)
    document.getElementById('toggle-mobius').addEventListener('click', () => {
        alert('Möbius toggle requires backend integration.\n' +
              'Run: python export_echozero_state.py\n' +
              'with Möbius layer enabled.');
    });

    // Inject hallucination (placeholder)
    document.getElementById('inject-hallucination').addEventListener('click', () => {
        alert('Hallucination injection requires backend integration.\n' +
              'Modify export_echozero_state.py to inject test patterns.');
    });

    // Spacebar to pause/resume
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            paused = !paused;
            document.getElementById('status-text').textContent =
                paused ? 'Paused' : 'Live';
        }
    });
}

/**
 * Toggle recording
 */
function toggleRecording() {
    const btn = document.getElementById('record');

    if (!recording) {
        // Start recording
        const stream = renderer.domElement.captureStream(30);
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `echozero_${Date.now()}.webm`;
            a.click();
            recordedChunks = [];
        };

        recordedChunks = [];
        mediaRecorder.start();
        recording = true;
        btn.textContent = '⏹️ Stop Recording';
        btn.classList.add('recording');
    } else {
        // Stop recording
        mediaRecorder.stop();
        recording = false;
        btn.textContent = '🔴 Record MP4';
        btn.classList.remove('recording');
    }
}

/**
 * Initialize application
 */
async function init() {
    console.log('🌿 EchoZero Vine Memory Visualization');
    console.log('Initializing...');

    // Initialize scene
    initScene();
    initGraph();
    setupUI();

    // Hide loading, show UI
    document.getElementById('loading').style.display = 'none';
    document.getElementById('ui-panel').style.display = 'block';
    document.getElementById('status').style.display = 'flex';
    document.getElementById('graph-container').style.display = 'block';
    document.getElementById('help').style.display = 'block';

    // Start animation loop
    animate();

    // Start state update loop
    stateLoop();

    console.log('✅ Visualization ready');
    console.log('Waiting for state updates from export_echozero_state.py...');
}

// Start application
init();
