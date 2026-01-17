"""
ROS2 Node for GRCM Integration
Ready-to-use node for Optimus robot deployment
"""
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from .utils import qualia_tensor_to_dict


@dataclass
class GRCMMessage:
    """ROS2-compatible message format for GRCM outputs"""
    phi: float
    coherence: float
    conflict: float
    action_halted: bool
    halt_reason: str
    qualia_calm: float
    qualia_alert: float
    qualia_curious: float
    qualia_conflicted: float
    timestamp: float


class GRCMRos2Node:
    """
    ROS2 Node wrapper for GRCM
    
    Provides standard ROS2 interface for robotics applications.
    
    Topics Published:
    - /grcm/state: Current GRCM state (GRCMMessage)
    - /grcm/halt: Emergency halt signals
    - /grcm/qualia: Qualia state updates
    
    Topics Subscribed:
    - /sensors/camera: Camera embeddings
    - /sensors/lidar: Lidar embeddings
    - /sensors/proprioception: Body state
    
    Usage (requires rclpy):
        import rclpy
        from grcm import ModularGRCM
        from grcm.integrations import GRCMRos2Node
        
        rclpy.init()
        model = ModularGRCM()
        node = GRCMRos2Node(model, node_name='grcm_optimus')
        rclpy.spin(node.get_node())
    
    Note: This is a template. Actual ROS2 integration requires rclpy
    which is only available in ROS2 environments.
    """
    
    def __init__(
        self,
        model,
        node_name: str = 'grcm_node',
        namespace: str = '',
        phi_threshold: float = 0.3,
        conflict_threshold: float = 0.5
    ):
        """
        Initialize ROS2 GRCM node
        
        Args:
            model: ModularGRCM instance
            node_name: ROS2 node name
            namespace: Optional namespace prefix
            phi_threshold: Threshold for phi gating
            conflict_threshold: Threshold for conflict detection
        """
        self.model = model
        self.node_name = node_name
        self.namespace = namespace
        self.phi_threshold = phi_threshold
        self.conflict_threshold = conflict_threshold
        
        self._node = None
        self._publishers = {}
        self._subscribers = {}
        self._timer = None
        
        self._latest_camera = None
        self._latest_lidar = None
        self._latest_proprio = None
        
    def init_ros2(self):
        """
        Initialize ROS2 components
        
        Call this method after rclpy.init()
        """
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float32MultiArray, Bool, String
            
            class _GRCMNode(Node):
                def __init__(node_self, name):
                    super().__init__(name)
                    self._setup_node(node_self)
                    
            self._node = _GRCMNode(self.node_name)
            return True
            
        except ImportError:
            print("Warning: rclpy not available. Running in simulation mode.")
            return False
            
    def _setup_node(self, node):
        """Setup publishers, subscribers, and timers"""
        from std_msgs.msg import Float32MultiArray, Bool
        
        self._publishers['state'] = node.create_publisher(
            Float32MultiArray, 
            f'{self.namespace}/grcm/state', 
            10
        )
        self._publishers['halt'] = node.create_publisher(
            Bool,
            f'{self.namespace}/grcm/halt',
            10
        )
        
        self._subscribers['camera'] = node.create_subscription(
            Float32MultiArray,
            f'{self.namespace}/sensors/camera',
            self._camera_callback,
            10
        )
        
        self._timer = node.create_timer(0.1, self._inference_callback)
        
    def _camera_callback(self, msg):
        """Handle incoming camera data"""
        self._latest_camera = torch.tensor(msg.data).unsqueeze(0)
        
    def _inference_callback(self):
        """Run inference and publish results"""
        if self._latest_camera is None:
            return
            
        audio_emb = self._latest_lidar if self._latest_lidar is not None \
                    else torch.zeros(1, 768)
                    
        outputs = self.model(self._latest_camera, audio_emb)
        
        phi = float(outputs['phi'])
        qualia = qualia_tensor_to_dict(outputs['qualia'])
        conflict = float(qualia.get('conflicted', 0))
        
        should_halt = phi < self.phi_threshold or conflict > self.conflict_threshold
        
        if should_halt and 'halt' in self._publishers:
            from std_msgs.msg import Bool
            halt_msg = Bool()
            halt_msg.data = True
            self._publishers['halt'].publish(halt_msg)
            
    def get_node(self):
        """Get the ROS2 node for spinning"""
        return self._node
        
    def process_sync(
        self,
        camera_data: torch.Tensor,
        lidar_data: Optional[torch.Tensor] = None
    ) -> GRCMMessage:
        """
        Synchronous processing (for non-ROS2 testing)
        
        Args:
            camera_data: Camera embeddings
            lidar_data: Optional lidar embeddings
            
        Returns:
            GRCMMessage with results
        """
        audio_emb = lidar_data if lidar_data is not None else torch.zeros(1, 768)
        outputs = self.model(camera_data, audio_emb)
        
        phi = float(outputs['phi'])
        coherence = float(outputs['coherence'].mean())
        qualia = qualia_tensor_to_dict(outputs['qualia'])
        conflict = float(qualia.get('conflicted', 0))
        
        action_halted = phi < self.phi_threshold or conflict > self.conflict_threshold
        halt_reason = ""
        if phi < self.phi_threshold:
            halt_reason = f"Low phi ({phi:.2f})"
        elif conflict > self.conflict_threshold:
            halt_reason = f"High conflict ({conflict:.2f})"
            
        return GRCMMessage(
            phi=phi,
            coherence=coherence,
            conflict=conflict,
            action_halted=action_halted,
            halt_reason=halt_reason,
            qualia_calm=float(qualia.get('calm', 0)),
            qualia_alert=float(qualia.get('alert', 0)),
            qualia_curious=float(qualia.get('curious', 0)),
            qualia_conflicted=conflict,
            timestamp=float(outputs.get('timestamp', 0))
        )


def create_launch_file(package_name: str = 'grcm_robot') -> str:
    """
    Generate ROS2 launch file content
    
    Args:
        package_name: Name of the ROS2 package
        
    Returns:
        Python launch file content as string
    """
    return f'''"""
ROS2 Launch file for GRCM node
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='grcm_node',
            name='grcm_consciousness',
            parameters=[{{
                'phi_threshold': 0.3,
                'conflict_threshold': 0.5,
                'inference_rate': 10.0
            }}],
            remappings=[
                ('/sensors/camera', '/optimus/camera/embeddings'),
                ('/sensors/lidar', '/optimus/lidar/embeddings'),
                ('/grcm/halt', '/optimus/emergency_stop')
            ]
        )
    ])
'''
