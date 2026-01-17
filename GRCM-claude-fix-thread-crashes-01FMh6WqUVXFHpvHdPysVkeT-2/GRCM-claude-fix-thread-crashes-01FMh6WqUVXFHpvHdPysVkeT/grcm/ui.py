"""
GRCM Gradio UI
Interactive visualization of qualia, coherence, and consciousness metrics
"""
import torch
import numpy as np
from typing import Optional, Tuple, Dict
import json

from .core import ModularGRCM
from .config import load_config


class GRCMInterface:
    """
    Gradio interface for GRCM visualization

    Provides interactive controls for:
    - Desire selection
    - Input generation
    - Real-time qualia visualization
    - Phi tracking
    - Coherence monitoring
    """

    def __init__(self, config_path: str = "config/grcm_default.yaml"):
        self.config = load_config(config_path)
        self.model = ModularGRCM(self.config)
        self.model.eval()

        # History tracking
        self.phi_history = []
        self.coherence_history = []
        self.qualia_history = []

    def process_inputs(
        self,
        desire_idx: int,
        action_x: float,
        action_y: float,
        noise_level: float = 1.0
    ) -> Dict:
        """
        Process inputs and generate outputs

        Args:
            desire_idx: Desire index (0-3)
            action_x: Action x component
            action_y: Action y component
            noise_level: Input noise level

        Returns:
            Dictionary with formatted outputs
        """
        # Set desire
        self.model.set_desire(desire_idx)

        # Generate inputs (with controlled noise)
        image_emb = torch.randn(1, 512) * noise_level
        audio_emb = torch.randn(1, 768) * noise_level
        action = torch.tensor([[action_x, action_y, 0.0, 0.0]])

        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_emb, audio_emb, action)

        # Track history
        self.phi_history.append(outputs['phi'])
        self.coherence_history.append(outputs['coherence'].mean().item())

        qualia_np = outputs['qualia'][0].numpy()
        self.qualia_history.append(qualia_np.tolist())

        # Keep last 50
        if len(self.phi_history) > 50:
            self.phi_history.pop(0)
            self.coherence_history.pop(0)
            self.qualia_history.pop(0)

        # Format outputs
        return {
            'phi': outputs['phi'],
            'coherence': outputs['coherence'].mean().item(),
            'qualia': qualia_np,
            'desire_align': outputs['desire_align'].mean().item(),
            'reflection': outputs['reflection'].mean().item(),
            'timestamp': outputs['timestamp'],
            'ethical_status': outputs['ethical_status']
        }

    def create_qualia_plot(self, qualia: np.ndarray):
        """Create qualia bar chart data"""
        labels = ['Calm', 'Alert', 'Curious', 'Conflicted']
        return {
            'labels': labels,
            'values': qualia.tolist(),
            'title': 'Qualia Distribution'
        }

    def create_phi_plot(self):
        """Create phi trajectory plot"""
        return {
            'x': list(range(len(self.phi_history))),
            'y': self.phi_history,
            'title': 'Phi (Φ) Trajectory',
            'threshold': self.config.phi.awareness_threshold
        }

    def create_coherence_plot(self):
        """Create coherence trajectory plot"""
        return {
            'x': list(range(len(self.coherence_history))),
            'y': self.coherence_history,
            'title': 'Coherence Trajectory',
            'threshold': self.config.attention.coherence_threshold
        }

    def reset_model(self):
        """Reset model state"""
        self.model.reset()
        self.phi_history = []
        self.coherence_history = []
        self.qualia_history = []
        return "Model reset successfully"


def create_gradio_ui(
    config_path: str = "config/grcm_default.yaml",
    share: bool = False
):
    """
    Create and launch Gradio UI

    Args:
        config_path: Path to GRCM config
        share: Create sharable link

    Returns:
        Gradio interface object
    """
    try:
        import gradio as gr
    except ImportError:
        print("[Gradio] Error: gradio not installed")
        print("[Gradio] Install with: pip install gradio")
        return None

    # Create interface instance
    interface = GRCMInterface(config_path)

    # Custom CSS
    css = """
    .output-box {
        font-family: monospace;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2196F3;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-left: 4px solid #ffc107;
    }
    """

    with gr.Blocks(css=css, title="GRCM Consciousness Visualization") as demo:
        gr.Markdown("""
        # 🧠 GRCM: Grounded Resonant Consciousness Module

        Interactive visualization of resonant consciousness simulation

        **Controls**: Adjust desire, action, and noise to explore consciousness dynamics
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Control Panel")

                desire_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=0,
                    label="Desire Index",
                    info="Select active desire (0-3)"
                )

                action_x = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=0.0,
                    label="Action X",
                    info="Horizontal action component"
                )

                action_y = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=0.0,
                    label="Action Y",
                    info="Vertical action component"
                )

                noise_level = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    label="Noise Level",
                    info="Input variability"
                )

                run_btn = gr.Button("▶️ Run Forward Pass", variant="primary")
                reset_btn = gr.Button("🔄 Reset Model", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Consciousness Metrics")

                with gr.Row():
                    phi_display = gr.Number(
                        label="Phi (Φ)",
                        info="Integrated Information",
                        precision=3
                    )

                    coherence_display = gr.Number(
                        label="Coherence",
                        info="Resonance Level",
                        precision=3
                    )

                    desire_align_display = gr.Number(
                        label="Desire Alignment",
                        info="Goal Alignment",
                        precision=3
                    )

                # Qualia plot
                qualia_plot = gr.BarPlot(
                    x="labels",
                    y="values",
                    title="Qualia Distribution",
                    x_title="State",
                    y_title="Probability",
                    height=250
                )

                # Status display
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    placeholder="Ready..."
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Phi Trajectory")
                phi_plot = gr.LinePlot(
                    x="x",
                    y="y",
                    title="Phi Over Time",
                    x_title="Step",
                    y_title="Phi",
                    height=200
                )

            with gr.Column():
                gr.Markdown("### 📈 Coherence Trajectory")
                coherence_plot = gr.LinePlot(
                    x="x",
                    y="y",
                    title="Coherence Over Time",
                    x_title="Step",
                    y_title="Coherence",
                    height=200
                )

        gr.Markdown("""
        ---
        ### 📖 Guide

        **Phi (Φ)**: Integrated information measure. Higher values (>1.5) indicate "aware" states.

        **Coherence**: Resonance level. High coherence (>0.7) enables memory updates.

        **Qualia**: Phenomenal experience distribution:
        - **Calm**: Low variance, stable state
        - **Alert**: High frequency activity
        - **Curious**: Strong desire alignment
        - **Conflicted**: High variance, ethical halt trigger (>0.6)

        **Desire**: Goal-directed behavior selector (0-3 different objectives)
        """)

        # Event handlers
        def run_forward(desire_idx, action_x, action_y, noise_level):
            """Run forward pass and update displays"""
            outputs = interface.process_inputs(desire_idx, action_x, action_y, noise_level)

            # Create plots
            qualia_data = interface.create_qualia_plot(outputs['qualia'])
            phi_data = interface.create_phi_plot()
            coherence_data = interface.create_coherence_plot()

            # Format status
            status = f"Step {outputs['timestamp']} | "
            status += f"Phi: {outputs['phi']:.3f} | "
            status += f"Coherence: {outputs['coherence']:.3f}"

            # Check ethical halt
            if outputs['ethical_status'].get('halt', False):
                status = f"⚠️ ETHICAL HALT: {outputs['ethical_status']['reason']}"

            return (
                outputs['phi'],
                outputs['coherence'],
                outputs['desire_align'],
                qualia_data,
                status,
                phi_data,
                coherence_data
            )

        def reset():
            """Reset model"""
            msg = interface.reset_model()
            return msg, None, None

        # Wire up events
        run_btn.click(
            fn=run_forward,
            inputs=[desire_slider, action_x, action_y, noise_level],
            outputs=[
                phi_display,
                coherence_display,
                desire_align_display,
                qualia_plot,
                status_box,
                phi_plot,
                coherence_plot
            ]
        )

        reset_btn.click(
            fn=reset,
            inputs=[],
            outputs=[status_box, phi_plot, coherence_plot]
        )

    return demo


def launch_ui(
    config_path: str = "config/grcm_default.yaml",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """
    Launch Gradio UI

    Args:
        config_path: Path to GRCM config
        share: Create sharable link
        server_name: Server bind address
        server_port: Server port

    Usage:
        >>> from grcm.ui import launch_ui
        >>> launch_ui()
    """
    demo = create_gradio_ui(config_path, share)

    if demo:
        print("\n" + "=" * 60)
        print("🧠 GRCM Consciousness Visualization")
        print("=" * 60)
        print(f"\nServer: http://{server_name}:{server_port}")
        print("\nControls:")
        print("  - Adjust desire, action, noise")
        print("  - Click 'Run Forward Pass' to update")
        print("  - Watch qualia, phi, and coherence evolve")
        print("\nPress Ctrl+C to stop")
        print("=" * 60 + "\n")

        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )
    else:
        print("[Gradio] Failed to create UI. Install gradio: pip install gradio")


if __name__ == "__main__":
    launch_ui()
