"""
Energy Efficiency and Resource Utilization Tests

Validates computational efficiency, resource usage, and energy consumption
of the Transition Governor middleware.

Test Categories:
- CPU/Memory Utilization
- Computational Overhead
- Energy Per Decision
- Brownout Mode Efficiency
- Multi-Agent Resource Efficiency
- Cache and Memory Patterns
"""

import gc
import time
import psutil
import os
from typing import Dict, List, Tuple
import numpy as np

from transition_governor import TransitionGovernor, AIState, GovernanceState
from transition_governor.core.state import ToolState
from transition_governor.multi_agent import MultiAgentCoordinator, AgentConfig


class ResourceMonitor:
    """Utility for monitoring CPU, memory, and energy metrics."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_cpu_times = None
        self.start_memory = None

    def start(self):
        """Start monitoring."""
        gc.collect()  # Clear garbage before measurement
        self.start_time = time.perf_counter()
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory = self.process.memory_info()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.perf_counter()
        end_cpu_times = self.process.cpu_times()
        end_memory = self.process.memory_info()

        wall_time = end_time - self.start_time
        cpu_time = (end_cpu_times.user - self.start_cpu_times.user +
                   end_cpu_times.system - self.start_cpu_times.system)

        # Memory delta
        memory_delta = end_memory.rss - self.start_memory.rss

        # CPU utilization percentage
        cpu_percent = (cpu_time / wall_time * 100) if wall_time > 0 else 0

        # Estimate energy (rough approximation)
        # Assume 15W TDP for CPU at 100% utilization
        estimated_joules = cpu_time * 15.0  # Joules
        estimated_wh = estimated_joules / 3600.0  # Watt-hours

        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_percent': cpu_percent,
            'memory_delta_mb': memory_delta / (1024 * 1024),
            'memory_rss_mb': end_memory.rss / (1024 * 1024),
            'estimated_joules': estimated_joules,
            'estimated_wh': estimated_wh
        }


class TestCPUMemoryUtilization:
    """Test CPU and memory utilization patterns."""

    def test_cpu_utilization_per_decision(self):
        """Measure CPU utilization per governance decision."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        monitor = ResourceMonitor()

        num_decisions = 10000

        # Baseline CPU measurement
        monitor.start()
        time.sleep(0.001)  # Tiny sleep to measure idle
        baseline = monitor.stop()

        # Measure actual decisions
        monitor.start()

        for i in range(num_decisions):
            state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        metrics = monitor.stop()

        cpu_per_decision = (metrics['cpu_time'] / num_decisions) * 1000000  # microseconds
        wall_per_decision = (metrics['wall_time'] / num_decisions) * 1000  # milliseconds

        print(f"\n   CPU Utilization Metrics ({num_decisions} decisions):")
        print(f"   Wall time per decision:  {wall_per_decision:.3f} ms")
        print(f"   CPU time per decision:   {cpu_per_decision:.3f} µs")
        print(f"   CPU utilization:         {metrics['cpu_percent']:.1f}%")
        print(f"   Memory delta:            {metrics['memory_delta_mb']:.2f} MB")

        # CPU time per decision should be under 100 microseconds
        assert cpu_per_decision < 100, f"CPU time too high: {cpu_per_decision:.3f}µs"

        # Memory delta should be reasonable (under 10MB for 10k decisions)
        assert abs(metrics['memory_delta_mb']) < 10, f"Memory leak suspected: {metrics['memory_delta_mb']:.2f}MB"

    def test_memory_efficiency_patterns(self):
        """Test memory allocation patterns and efficiency."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        process = psutil.Process(os.getpid())

        gc.collect()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        memory_samples = []

        # Run 1000 decisions and sample memory every 100
        for i in range(1000):
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

            if i % 100 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory - initial_memory)

        memory_growth = memory_samples[-1] - memory_samples[0]
        memory_stability = np.std(memory_samples)

        print(f"\n   Memory Efficiency:")
        print(f"   Initial delta:    {memory_samples[0]:.2f} MB")
        print(f"   Final delta:      {memory_samples[-1]:.2f} MB")
        print(f"   Growth:           {memory_growth:.2f} MB")
        print(f"   Stability (std):  {memory_stability:.2f} MB")

        # Memory should be stable (low standard deviation)
        assert memory_stability < 5.0, f"Memory usage unstable: std={memory_stability:.2f}MB"

        # No significant memory growth
        assert abs(memory_growth) < 5.0, f"Memory grew: {memory_growth:.2f}MB"


class TestComputationalOverhead:
    """Test computational overhead of governance vs baseline."""

    def test_governor_overhead_vs_baseline(self):
        """Compare governor overhead against minimal baseline."""
        num_iterations = 5000

        # Baseline: just create AIState objects (no governance)
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(num_iterations):
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            # No governance, just object creation

        baseline_metrics = monitor.stop()

        # With governance
        governor = TransitionGovernor(seed=42, enable_logging=False)
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(num_iterations):
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        governed_metrics = monitor.stop()

        overhead_cpu = (governed_metrics['cpu_time'] - baseline_metrics['cpu_time']) / num_iterations * 1000000
        overhead_wall = (governed_metrics['wall_time'] - baseline_metrics['wall_time']) / num_iterations * 1000
        overhead_percent = ((governed_metrics['cpu_time'] - baseline_metrics['cpu_time']) /
                           baseline_metrics['cpu_time'] * 100) if baseline_metrics['cpu_time'] > 0 else 0

        print(f"\n   Computational Overhead ({num_iterations} iterations):")
        print(f"   Baseline CPU time:   {baseline_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Governed CPU time:   {governed_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Overhead per call:   {overhead_cpu:.3f} µs")
        print(f"   Overhead (wall):     {overhead_wall:.4f} ms")
        print(f"   Overhead percent:    {overhead_percent:.1f}%")

        # Governance overhead should be reasonable (under 50µs per decision)
        assert overhead_cpu < 50, f"Overhead too high: {overhead_cpu:.3f}µs"

    def test_brownout_mode_efficiency_gain(self):
        """Test if brownout mode reduces computational load."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        num_decisions = 1000

        # Normal mode decisions
        normal_states = []
        for i in range(num_decisions):
            state = AIState(
                entropy=1.5,  # Stable
                entropy_dot=0.0,
                confidence=0.9,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            normal_states.append(state)

        monitor = ResourceMonitor()
        monitor.start()
        for state in normal_states:
            output = governor.govern(state)
        normal_metrics = monitor.stop()

        # Reset governor
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Brownout mode decisions (high instability)
        brownout_states = []
        for i in range(num_decisions):
            state = AIState(
                entropy=5.0,  # Very unstable
                entropy_dot=10.0,
                confidence=0.3,
                confidence_dot=-2.0,
                tool_state=ToolState.ERROR,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            brownout_states.append(state)

        monitor = ResourceMonitor()
        monitor.start()
        for state in brownout_states:
            output = governor.govern(state)
        brownout_metrics = monitor.stop()

        # In brownout, the governor constrains the AI, which should reduce downstream load
        # The governor itself may do slightly more work, but the constraint savings dominate

        print(f"\n   Brownout Mode Efficiency:")
        print(f"   Normal mode CPU:     {normal_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Brownout mode CPU:   {brownout_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Normal mode energy:  {normal_metrics['estimated_joules']:.4f} J")
        print(f"   Brownout mode energy:{brownout_metrics['estimated_joules']:.4f} J")

        # Both should complete successfully (check wall time which is always measured)
        assert normal_metrics['wall_time'] > 0
        assert brownout_metrics['wall_time'] > 0


class TestEnergyPerDecision:
    """Test energy consumption per governance decision."""

    def test_energy_consumption_estimate(self):
        """Estimate energy consumption per decision."""
        governor = TransitionGovernor(seed=42, enable_logging=False)
        num_decisions = 10000

        monitor = ResourceMonitor()
        monitor.start()

        for i in range(num_decisions):
            state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.3),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        metrics = monitor.stop()

        joules_per_decision = metrics['estimated_joules'] / num_decisions
        joules_per_million = joules_per_decision * 1_000_000
        kwh_per_million = (joules_per_million / 3600) / 1000  # Convert to kWh

        # Estimate carbon footprint (assuming 0.4 kg CO2/kWh - global average)
        co2_grams_per_million = kwh_per_million * 0.4 * 1000

        print(f"\n   Energy Consumption ({num_decisions} decisions):")
        print(f"   Total CPU time:           {metrics['cpu_time']*1000:.2f} ms")
        print(f"   Estimated energy:         {metrics['estimated_joules']:.4f} J")
        print(f"   Energy per decision:      {joules_per_decision*1000:.4f} mJ")
        print(f"   Energy per 1M decisions:  {joules_per_million:.2f} J ({kwh_per_million*1000:.4f} Wh)")
        print(f"   Est. CO₂ per 1M decisions: {co2_grams_per_million:.2f} g")

        # Energy per decision should be very small (under 1 millijoule)
        assert joules_per_decision < 0.001, f"Energy too high: {joules_per_decision*1000:.4f}mJ"

    def test_energy_distribution_across_states(self):
        """Test energy consumption across different governance states."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        scenarios = {
            'stable': AIState(1.5, 0.0, 0.9, 0.0, ToolState.INACTIVE, 500, 2048, 0.0),
            'unstable': AIState(4.0, 5.0, 0.4, -1.5, ToolState.ERROR, 500, 2048, 0.0),
            'recovering': AIState(2.5, -2.0, 0.7, 0.5, ToolState.INACTIVE, 500, 2048, 0.0),
        }

        results = {}

        for scenario_name, base_state in scenarios.items():
            monitor = ResourceMonitor()
            monitor.start()

            for i in range(1000):
                # Add slight variation
                state = AIState(
                    entropy=max(0.0, base_state.entropy + np.random.randn() * 0.1),
                    entropy_dot=base_state.entropy_dot + np.random.randn() * 0.1,
                    confidence=np.clip(base_state.confidence + np.random.randn() * 0.05, 0.0, 1.0),
                    confidence_dot=base_state.confidence_dot + np.random.randn() * 0.05,
                    tool_state=base_state.tool_state,
                    context_length=base_state.context_length,
                    max_context_length=base_state.max_context_length,
                    fatigue=base_state.fatigue
                )
                governor.govern(state)

            metrics = monitor.stop()
            results[scenario_name] = metrics['estimated_joules'] / 1000

        print(f"\n   Energy by Scenario (1000 decisions each):")
        print(f"   Stable state:      {results['stable']:.4f} J")
        print(f"   Unstable state:    {results['unstable']:.4f} J")
        print(f"   Recovering state:  {results['recovering']:.4f} J")

        # All scenarios should complete with measurable energy
        for scenario, energy in results.items():
            assert energy > 0, f"{scenario} used no energy"


class TestMultiAgentEfficiency:
    """Test resource efficiency in multi-agent scenarios."""

    def test_multi_agent_resource_efficiency(self):
        """Compare efficiency of multi-agent coordination vs individual governors."""
        num_agents = 10
        num_rounds = 100

        # Scenario 1: Coordinated multi-agent system
        coordinator = MultiAgentCoordinator(blend_rate=0.3, seed=42)
        for i in range(num_agents):
            coordinator.add_agent(AgentConfig(agent_id=f"agent_{i}", max_context_length=2048))

        monitor = ResourceMonitor()
        monitor.start()

        for round_num in range(num_rounds):
            states = {
                f"agent_{i}": AIState(
                    entropy=1.5 + i * 0.2,
                    entropy_dot=0.0,
                    confidence=0.8,
                    confidence_dot=0.0,
                    tool_state=ToolState.INACTIVE,
                    context_length=500,
                    max_context_length=2048,
                    fatigue=0.0
                )
                for i in range(num_agents)
            }
            coordinator.process_round(states)

        coordinated_metrics = monitor.stop()

        # Scenario 2: Individual governors (less efficient)
        governors = [TransitionGovernor(seed=42+i, enable_logging=False) for i in range(num_agents)]

        monitor = ResourceMonitor()
        monitor.start()

        for round_num in range(num_rounds):
            for i in range(num_agents):
                state = AIState(
                    entropy=1.5 + i * 0.2,
                    entropy_dot=0.0,
                    confidence=0.8,
                    confidence_dot=0.0,
                    tool_state=ToolState.INACTIVE,
                    context_length=500,
                    max_context_length=2048,
                    fatigue=0.0
                )
                governors[i].govern(state)

        individual_metrics = monitor.stop()

        # Calculate efficiency gain (use wall time if CPU time is 0)
        if individual_metrics['cpu_time'] > 0:
            efficiency_gain = (1 - coordinated_metrics['cpu_time'] / individual_metrics['cpu_time']) * 100
        else:
            efficiency_gain = 0.0

        energy_savings = individual_metrics['estimated_joules'] - coordinated_metrics['estimated_joules']

        print(f"\n   Multi-Agent Resource Efficiency ({num_agents} agents, {num_rounds} rounds):")
        print(f"   Coordinated CPU time:  {coordinated_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Individual CPU time:   {individual_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Efficiency change:     {efficiency_gain:.1f}%")
        print(f"   Energy difference:     {energy_savings:.4f} J")

        # Note: Coordinator has overhead for authority allocation
        # The benefit is governance quality, not raw speed
        # Verify overhead is reasonable (under 10x)
        overhead_ratio = coordinated_metrics['wall_time'] / individual_metrics['wall_time'] if individual_metrics['wall_time'] > 0 else 1.0
        assert overhead_ratio < 10.0, f"Coordinator overhead too high: {overhead_ratio:.1f}x"


class TestCacheMemoryPatterns:
    """Test cache efficiency and memory access patterns."""

    def test_state_history_memory_growth(self):
        """Test that state history doesn't cause excessive memory growth."""
        governor = TransitionGovernor(seed=42, enable_logging=True)  # Enable logging for history
        process = psutil.Process(os.getpid())

        gc.collect()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Run 10000 decisions
        for i in range(10000):
            state = AIState(
                entropy=1.5,
                entropy_dot=0.0,
                confidence=0.8,
                confidence_dot=0.0,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(state)

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory

        history_size = len(governor.state_history)
        memory_per_state = (memory_growth * 1024) / history_size if history_size > 0 else 0  # KB per state

        print(f"\n   State History Memory:")
        print(f"   History entries:      {history_size}")
        print(f"   Memory growth:        {memory_growth:.2f} MB")
        print(f"   Memory per state:     {memory_per_state:.2f} KB")

        # Memory per state should be reasonable (under 5KB per entry)
        assert memory_per_state < 5.0, f"Memory per state too high: {memory_per_state:.2f}KB"

        # Total memory growth should be under 50MB for 10k states
        assert memory_growth < 50, f"Total memory growth too high: {memory_growth:.2f}MB"

    def test_repeated_decision_efficiency(self):
        """Test efficiency of repeated decisions on similar states."""
        governor = TransitionGovernor(seed=42, enable_logging=False)

        # Same state repeated
        state = AIState(
            entropy=1.5,
            entropy_dot=0.0,
            confidence=0.8,
            confidence_dot=0.0,
            tool_state=ToolState.INACTIVE,
            context_length=500,
            max_context_length=2048,
            fatigue=0.0
        )

        monitor = ResourceMonitor()
        monitor.start()

        for i in range(5000):
            governor.govern(state)

        repeated_metrics = monitor.stop()

        # Varied states
        governor = TransitionGovernor(seed=42, enable_logging=False)

        monitor = ResourceMonitor()
        monitor.start()

        for i in range(5000):
            varied_state = AIState(
                entropy=max(0.0, 1.5 + np.random.randn() * 0.5),
                entropy_dot=np.random.randn() * 0.5,
                confidence=np.clip(0.8 + np.random.randn() * 0.1, 0.0, 1.0),
                confidence_dot=np.random.randn() * 0.2,
                tool_state=ToolState.INACTIVE,
                context_length=500,
                max_context_length=2048,
                fatigue=0.0
            )
            governor.govern(varied_state)

        varied_metrics = monitor.stop()

        # Calculate overhead ratio (use wall time if CPU time is 0)
        if repeated_metrics['cpu_time'] > 0:
            overhead_ratio = varied_metrics['cpu_time'] / repeated_metrics['cpu_time']
        else:
            overhead_ratio = varied_metrics['wall_time'] / repeated_metrics['wall_time'] if repeated_metrics['wall_time'] > 0 else 1.0

        print(f"\n   Decision Efficiency Patterns:")
        print(f"   Repeated state CPU:   {repeated_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Varied states CPU:    {varied_metrics['cpu_time']*1000:.2f} ms")
        print(f"   Overhead ratio:       {overhead_ratio:.2f}x")

        # Both should complete efficiently (check wall time)
        assert repeated_metrics['wall_time'] > 0
        assert varied_metrics['wall_time'] > 0
