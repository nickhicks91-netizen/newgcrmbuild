"""
Tesla FSD Integration Example

Demonstrates GRCM integration for autonomous driving perception pipeline.
This example shows how to use phi gating and ethical veto for safe decisions.
"""
import torch
import sys
sys.path.insert(0, '..')

from grcm import ModularGRCM, GRCMConfig
from grcm.integrations import TeslaPerception


def main():
    print("=" * 60)
    print("GRCM + Tesla FSD Integration Demo")
    print("=" * 60)
    
    config = GRCMConfig(
        input_dim=512,
        freq_dim=256,
        memory_size=128,
        enable_ethical_halt=True
    )
    model = ModularGRCM(config)
    perception = TeslaPerception(model)
    
    print("\n[Scenario 1: Clear Road - High Confidence]")
    print("-" * 40)
    
    camera_clear = torch.randn(1, 512) * 0.3 + 0.7
    lidar_clear = torch.randn(1, 768) * 0.3 + 0.7
    
    result = perception.process({
        'front_camera': camera_clear,
        'lidar_points': lidar_clear
    })
    
    print(f"Action: {result.action}")
    print(f"Phi (integration): {result.phi:.3f}")
    print(f"Conflict level: {result.conflict_level:.3f}")
    print(f"Action halted: {result.action_halted}")
    
    print("\n[Scenario 2: Conflicting Sensors]")
    print("-" * 40)
    
    camera_conflict = torch.randn(1, 512) * 2.0
    lidar_conflict = -camera_conflict[:, :768]
    
    result = perception.process({
        'front_camera': camera_conflict,
        'lidar_points': lidar_conflict.expand(-1, 768)
    })
    
    print(f"Action: {result.action}")
    print(f"Phi (integration): {result.phi:.3f}")
    print(f"Conflict level: {result.conflict_level:.3f}")
    print(f"Action halted: {result.action_halted}")
    if result.halt_reason:
        print(f"Halt reason: {result.halt_reason}")
        
    print("\n[Scenario 3: Lane Change Safety Check]")
    print("-" * 40)
    
    current_sensors = {
        'front_camera': torch.randn(1, 512),
        'lidar_points': torch.randn(1, 768),
        'radar_objects': torch.randn(1, 512)
    }
    
    safety = perception.check_maneuver_safety('lane_change', current_sensors)
    
    print(f"Maneuver: {safety['maneuver']}")
    print(f"Safe: {safety['safe']}")
    print(f"Recommendation: {safety['recommendation']}")
    print(f"Phi: {safety['phi']:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo complete. GRCM provides:")
    print("1. Phi gating - blocks uncertain decisions")
    print("2. Conflict detection - catches sensor disagreement")
    print("3. Maneuver safety - validates high-risk actions")
    print("=" * 60)


if __name__ == '__main__':
    main()
