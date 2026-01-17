"""
Streamlit dashboard for EchoZero + GRCM visualization.

Run with: streamlit run grcm/ui/resonance_dashboard.py
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..hybrid import EchoGRCMHybrid
from ..train import MultimodalDatastream


# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = {
        "phi": [],
        "coherence": [],
        "qualia": [],
    }


def initialize_model(n_nodes: int):
    """Initialize EchoGRCM model."""
    model = EchoGRCMHybrid(n_nodes=n_nodes)
    return model


def plot_resonance_state(psi: torch.Tensor):
    """Plot resonant state visualization."""
    psi_np = psi.cpu().numpy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Amplitude", "Phase"],
    )

    # Amplitude
    fig.add_trace(
        go.Scatter(
            y=np.abs(psi_np),
            mode='lines+markers',
            name='|ψ|',
            line=dict(color='blue'),
        ),
        row=1, col=1,
    )

    # Phase
    fig.add_trace(
        go.Scatter(
            y=np.angle(psi_np),
            mode='lines+markers',
            name='arg(ψ)',
            line=dict(color='red'),
        ),
        row=1, col=2,
    )

    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_coherence(coherence: torch.Tensor):
    """Plot coherence heatmap."""
    coherence_np = coherence.cpu().numpy()

    fig = px.line(
        y=coherence_np,
        title="Attention Coherence",
        labels={"y": "Coherence", "index": "Node"},
    )
    fig.update_layout(height=300)
    return fig


def plot_qualia(qualia: torch.Tensor):
    """Plot qualia distribution."""
    qualia_np = qualia.cpu().numpy()
    labels = ["Calm", "Alert", "Curious", "Conflicted"]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=qualia_np,
            marker=dict(color=['green', 'orange', 'blue', 'red']),
        )
    ])

    fig.update_layout(
        title="Qualia States",
        yaxis_title="Probability",
        height=300,
    )
    return fig


def main():
    """Main dashboard app."""
    st.set_page_config(
        page_title="EchoZero Resonance Dashboard",
        page_icon="🌀",
        layout="wide",
    )

    st.title("🌀 EchoZero + GRCM Resonance Dashboard")
    st.markdown("Real-time visualization of resonant intelligence dynamics")

    # Sidebar controls
    st.sidebar.header("System Configuration")

    n_nodes = st.sidebar.slider("Number of Nodes", 16, 256, 64, step=16)

    if st.sidebar.button("Initialize System") or st.session_state.model is None:
        with st.spinner("Initializing EchoZero system..."):
            st.session_state.model = initialize_model(n_nodes)
        st.sidebar.success("System initialized!")

    if st.session_state.model is None:
        st.warning("Please initialize the system from the sidebar")
        return

    model = st.session_state.model

    # Control panel
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Run Step"):
            # Generate random input
            datastream = MultimodalDatastream(batch_size=1)
            batch = datastream.generate_batch()

            # Forward pass
            with torch.no_grad():
                outputs = model(**batch)

            # Store outputs
            st.session_state.outputs = outputs

            # Update history
            st.session_state.history["phi"].append(outputs["phi"][0].item())
            st.session_state.history["coherence"].append(
                outputs["coherence"][0].mean().item()
            )
            st.session_state.history["qualia"].append(
                outputs["qualia"][0].tolist()
            )

    with col2:
        if st.button("🔄 Reset"):
            model.reset_state()
            st.session_state.history = {"phi": [], "coherence": [], "qualia": []}
            st.rerun()

    with col3:
        if st.button("💾 Save State"):
            # TODO: Implement state saving
            st.info("State saving not implemented yet")

    # Display results
    if "outputs" in st.session_state:
        outputs = st.session_state.outputs

        # Metrics row
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Φ (Integrated Info)", f"{outputs['phi'][0].item():.3f}")

        with metric_cols[1]:
            st.metric("Mean Coherence", f"{outputs['coherence'][0].mean().item():.3f}")

        with metric_cols[2]:
            dominant_qualia = outputs['qualia'][0].argmax().item()
            qualia_names = ["Calm", "Alert", "Curious", "Conflicted"]
            st.metric("Dominant Qualia", qualia_names[dominant_qualia])

        with metric_cols[3]:
            st.metric("Coupling Norm", f"{model.echozero.K.abs().mean().item():.3f}")

        # Visualizations
        st.subheader("Resonant State ψ")
        fig_psi = plot_resonance_state(outputs["psi"][0])
        st.plotly_chart(fig_psi, use_container_width=True)

        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.subheader("Coherence")
            fig_coherence = plot_coherence(outputs["coherence"][0])
            st.plotly_chart(fig_coherence, use_container_width=True)

        with col_viz2:
            st.subheader("Qualia")
            fig_qualia = plot_qualia(outputs["qualia"][0])
            st.plotly_chart(fig_qualia, use_container_width=True)

        # History plots
        if len(st.session_state.history["phi"]) > 1:
            st.subheader("Historical Trends")

            col_hist1, col_hist2 = st.columns(2)

            with col_hist1:
                fig_phi_hist = px.line(
                    y=st.session_state.history["phi"],
                    title="Φ over time",
                    labels={"y": "Φ", "index": "Step"},
                )
                st.plotly_chart(fig_phi_hist, use_container_width=True)

            with col_hist2:
                fig_coh_hist = px.line(
                    y=st.session_state.history["coherence"],
                    title="Mean Coherence over time",
                    labels={"y": "Coherence", "index": "Step"},
                )
                st.plotly_chart(fig_coh_hist, use_container_width=True)

    # System info
    with st.expander("System Information"):
        info = model.get_system_info()
        st.json(info)


if __name__ == "__main__":
    main()
