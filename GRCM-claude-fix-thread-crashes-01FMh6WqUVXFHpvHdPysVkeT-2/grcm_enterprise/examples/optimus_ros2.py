"""
Optimus Robot ROS2 Integration Example

Demonstrates GRCM integration for humanoid robot control.
Shows phi gating and conflict detection for safe robot actions.
"""
import torch
import sys
sys.path.insert(0, '..')

from grcm import ModularGRCM, GRCMConfig
from grcm.integrations import GRCMRos2Node, create_launch_file


def main():
    print("=" * 60)
    print("GRCM + Optimus ROS2 Integration Demo")
    print("=" * 60)
    
    config = GRCMConfig(
        input_dim=512,
        freq_dim=256,
        memory_size=128,
        enable_ethical_halt=True
    )
    model = ModularGRCM(config)
    
    ros_node = GRCMRos2Node(
        model,
        node_name='grcm_optimus',
        phi_threshold=0.3,
        conflict_threshold=0.5
    )
    
    print("\n[Simulating Robot Sensor Processing]")
    print("-" * 40)
    
    print("\nScenario 1: Normal operation")
    camera_normal = torch.randn(1, 512) * 0.5 + 0.5
    lidar_normal = torch.randn(1, 768) * 0.5 + 0.5
    
    result = ros_node.process_sync(camera_normal, lidar_normal)
    
    print(f"  Phi: {result.phi:.3f}")
    print(f"  Coherence: {result.coherence:.3f}")
    print(f"  Conflict: {result.conflict:.3f}")
    print(f"  Action halted: {result.action_halted}")
    print(f"  Qualia state: calm={result.qualia_calm:.2f}, alert={result.qualia_alert:.2f}")
    
    print("\nScenario 2: Uncertain sensor data")
    camera_noisy = torch.randn(1, 512) * 3.0
    lidar_noisy = torch.randn(1, 768) * 3.0
    
    result = ros_node.process_sync(camera_noisy, lidar_noisy)
    
    print(f"  Phi: {result.phi:.3f}")
    print(f"  Coherence: {result.coherence:.3f}")
    print(f"  Conflict: {result.conflict:.3f}")
    print(f"  Action halted: {result.action_halted}")
    if result.halt_reason:
        print(f"  Halt reason: {result.halt_reason}")
        
    print("\n[ROS2 Launch File]")
    print("-" * 40)
    
    launch_content = create_launch_file('grcm_optimus')
    print(launch_content[:500] + "...")
    
    print("\n[ROS2 Topics]")
    print("-" * 40)
    print("Published:")
    print("  /grcm/state     - Current GRCM state")
    print("  /grcm/halt      - Emergency halt signals")
    print("\nSubscribed:")
    print("  /sensors/camera - Camera embeddings")
    print("  /sensors/lidar  - Lidar embeddings")
    
    print("\n" + "=" * 60)
    print("To run in ROS2 environment:")
    print("  1. Install: pip install grcm[ros2]")
    print("  2. Build: colcon build --packages-select grcm_optimus")
    print("  3. Run: ros2 launch grcm_optimus grcm.launch.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
