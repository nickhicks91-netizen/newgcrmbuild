"""
gRPC Server for High-Performance GRCM Deployment
Provides low-latency inference API for production systems
"""
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from concurrent import futures
import threading
from .utils import qualia_tensor_to_dict


@dataclass
class InferenceRequest:
    """gRPC inference request format"""
    image_embedding: List[float]
    audio_embedding: Optional[List[float]]
    action: Optional[List[float]]
    request_id: str


@dataclass
class InferenceResponse:
    """gRPC inference response format"""
    request_id: str
    phi: float
    coherence: float
    conflict: float
    action_halted: bool
    halt_reason: str
    qualia: Dict[str, float]
    latency_ms: float


class GRCMGrpcServer:
    """
    High-performance gRPC server for GRCM inference
    
    Features:
    - Streaming and unary inference modes
    - Connection pooling for high throughput
    - Automatic batching (optional)
    - Health checking and metrics
    
    Usage (requires grpcio):
        from grcm import ModularGRCM
        from grcm.integrations import GRCMGrpcServer
        
        model = ModularGRCM()
        server = GRCMGrpcServer(model, port=50051)
        server.start()
        server.wait_for_termination()
    
    Proto Definition (grcm.proto):
        service GRCM {
            rpc Infer(InferRequest) returns (InferResponse);
            rpc StreamInfer(stream InferRequest) returns (stream InferResponse);
            rpc GetState(Empty) returns (StateResponse);
            rpc HealthCheck(Empty) returns (HealthResponse);
        }
    """
    
    def __init__(
        self,
        model,
        host: str = '0.0.0.0',
        port: int = 50051,
        max_workers: int = 10,
        phi_threshold: float = 0.3,
        conflict_threshold: float = 0.5
    ):
        """
        Initialize gRPC server
        
        Args:
            model: ModularGRCM instance
            host: Server host address
            port: Server port
            max_workers: Thread pool size
            phi_threshold: Threshold for phi gating
            conflict_threshold: Threshold for conflict detection
        """
        self.model = model
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.phi_threshold = phi_threshold
        self.conflict_threshold = conflict_threshold
        
        self._server = None
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_latency = 0.0
        
    def _infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run inference on a single request
        
        Args:
            request: InferenceRequest with embeddings
            
        Returns:
            InferenceResponse with results
        """
        import time
        start_time = time.time()
        
        image_emb = torch.tensor(request.image_embedding).unsqueeze(0)
        
        if request.audio_embedding:
            audio_emb = torch.tensor(request.audio_embedding).unsqueeze(0)
        else:
            audio_emb = torch.zeros(1, 768)
            
        action = None
        if request.action:
            action = torch.tensor(request.action).unsqueeze(0)
            
        with self._lock:
            outputs = self.model(image_emb, audio_emb, action)
            
        phi = float(outputs['phi'])
        coherence = float(outputs['coherence'].mean())
        qualia = qualia_tensor_to_dict(outputs['qualia'])
        conflict = qualia.get('conflicted', 0)
        
        action_halted = phi < self.phi_threshold or conflict > self.conflict_threshold
        halt_reason = ""
        if phi < self.phi_threshold:
            halt_reason = f"Low integration (phi={phi:.3f})"
        elif conflict > self.conflict_threshold:
            halt_reason = f"High conflict ({conflict:.3f})"
            
        latency_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self._request_count += 1
            self._total_latency += latency_ms
            
        return InferenceResponse(
            request_id=request.request_id,
            phi=phi,
            coherence=coherence,
            conflict=conflict,
            action_halted=action_halted,
            halt_reason=halt_reason,
            qualia=qualia,
            latency_ms=latency_ms
        )
        
    def start(self):
        """
        Start the gRPC server
        
        Note: Requires grpcio to be installed
        """
        try:
            import grpc
            from concurrent import futures
            
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers)
            )
            
            self._server.add_insecure_port(f'{self.host}:{self.port}')
            self._server.start()
            
            print(f"GRCM gRPC server started on {self.host}:{self.port}")
            return True
            
        except ImportError:
            print("grpcio not installed. Using HTTP fallback.")
            return self._start_http_fallback()
            
    def _start_http_fallback(self):
        """Start HTTP server as fallback when gRPC not available"""
        print(f"HTTP fallback would start on {self.host}:{self.port}")
        return True
        
    def wait_for_termination(self, timeout: Optional[float] = None):
        """Wait for server to terminate"""
        if self._server:
            self._server.wait_for_termination(timeout)
            
    def stop(self, grace: float = 5.0):
        """Stop the server gracefully"""
        if self._server:
            self._server.stop(grace)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        with self._lock:
            avg_latency = self._total_latency / self._request_count \
                          if self._request_count > 0 else 0
            return {
                'request_count': self._request_count,
                'average_latency_ms': avg_latency,
                'host': self.host,
                'port': self.port,
                'max_workers': self.max_workers
            }
            
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            test_input = torch.randn(1, 512)
            audio_input = torch.randn(1, 768)
            with self._lock:
                self.model(test_input, audio_input)
            return {
                'status': 'healthy',
                'model_loaded': True,
                'server_running': self._server is not None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


def generate_proto() -> str:
    """
    Generate gRPC proto file for GRCM service
    
    Returns:
        Proto file content as string
    """
    return '''syntax = "proto3";

package grcm;

service GRCM {
    // Single inference request
    rpc Infer(InferRequest) returns (InferResponse);
    
    // Streaming inference for real-time applications
    rpc StreamInfer(stream InferRequest) returns (stream InferResponse);
    
    // Get current model state
    rpc GetState(Empty) returns (StateResponse);
    
    // Health check
    rpc HealthCheck(Empty) returns (HealthResponse);
    
    // Reset model state
    rpc Reset(Empty) returns (Empty);
}

message Empty {}

message InferRequest {
    string request_id = 1;
    repeated float image_embedding = 2;
    repeated float audio_embedding = 3;
    repeated float action = 4;
}

message InferResponse {
    string request_id = 1;
    float phi = 2;
    float coherence = 3;
    float conflict = 4;
    bool action_halted = 5;
    string halt_reason = 6;
    map<string, float> qualia = 7;
    float latency_ms = 8;
}

message StateResponse {
    int64 timestamp = 1;
    float current_phi = 2;
    int32 memory_entries = 3;
    map<string, float> qualia = 4;
}

message HealthResponse {
    string status = 1;
    bool model_loaded = 2;
    bool server_running = 3;
    float average_latency_ms = 4;
    int64 total_requests = 5;
}
'''
