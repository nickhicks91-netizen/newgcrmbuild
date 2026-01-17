# GRCM Deployment Guide

Complete guide for deploying GRCM (Grounded Resonant Consciousness Module) to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Scaling](#scaling)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.24+ (for K8s deployment)
- **kubectl**: 1.24+
- **Python**: 3.10+ (for local development)
- **Git**: 2.30+

### Required Accounts
- GitHub account (for CI/CD)
- Container registry access (GHCR, DockerHub, etc.)
- Kubernetes cluster (GKE, EKS, AKS, or local Minikube)

### System Requirements
- **Minimum**: 2 CPU cores, 4GB RAM, 10GB disk
- **Recommended**: 4 CPU cores, 8GB RAM, 50GB disk
- **Production**: 8+ CPU cores, 16GB+ RAM, 100GB+ disk

---

## Docker Deployment

### Quick Start with Docker Compose

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/grcm.git
cd grcm
```

**2. Build and start services:**
```bash
docker-compose up -d
```

**3. Verify services are running:**
```bash
docker-compose ps
```

**4. Access services:**
- **GRCM API**: http://localhost:8000
- **Gradio UI**: http://localhost:7860
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Docker Compose Services

The `docker-compose.yml` includes 6 services:

1. **grcm-api** - BentoML REST API for inference
2. **grcm-ui** - Gradio interactive interface
3. **mlflow** - Experiment tracking and model registry
4. **prometheus** - Metrics collection
5. **grafana** - Monitoring dashboards
6. **redis** - Caching and job queue

### Building Custom Docker Images

**Build the base image:**
```bash
docker build -t grcm:latest -f Dockerfile .
```

**Build with specific Python version:**
```bash
docker build --build-arg PYTHON_VERSION=3.11 -t grcm:py311 .
```

**Multi-architecture build:**
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t grcm:latest .
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# GRCM Configuration
GRCM_CONFIG_PATH=config/grcm_default.yaml
GRCM_LOG_LEVEL=INFO
GRCM_DEVICE=cpu

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=GRCM-Production

# Prometheus Configuration
PROMETHEUS_RETENTION_TIME=30d

# Grafana Configuration
GF_SECURITY_ADMIN_PASSWORD=your_secure_password
GF_INSTALL_PLUGINS=grafana-piechart-panel

# Redis Configuration
REDIS_PASSWORD=your_redis_password
```

### Docker Commands

**View logs:**
```bash
docker-compose logs -f grcm-api
docker-compose logs -f grcm-ui
```

**Restart a service:**
```bash
docker-compose restart grcm-api
```

**Scale API instances:**
```bash
docker-compose up -d --scale grcm-api=3
```

**Stop all services:**
```bash
docker-compose down
```

**Clean up volumes:**
```bash
docker-compose down -v
```

---

## Kubernetes Deployment

### Prerequisites

**1. Set up kubectl context:**
```bash
kubectl config get-contexts
kubectl config use-context your-cluster
```

**2. Create namespace:**
```bash
kubectl create namespace grcm
kubectl config set-context --current --namespace=grcm
```

### Deploy to Kubernetes

**1. Apply ConfigMap:**
```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

**2. Deploy the application:**
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

**3. Create service and ingress:**
```bash
kubectl apply -f deployment/kubernetes/service.yaml
```

**4. Verify deployment:**
```bash
kubectl get pods
kubectl get services
kubectl get ingress
```

### Kubernetes Architecture

The deployment includes:

- **Deployment**: 3 replica pods with resource limits
- **HorizontalPodAutoscaler**: Auto-scaling 2-10 replicas
- **Service**: LoadBalancer for external access
- **Ingress**: HTTPS with cert-manager
- **ConfigMap**: Configuration files
- **Secret**: Sensitive credentials
- **PersistentVolumeClaim**: Model storage

### Scaling Configuration

**Manual scaling:**
```bash
kubectl scale deployment grcm-api --replicas=5
```

**HPA configuration:**
```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

**View autoscaling status:**
```bash
kubectl get hpa
kubectl describe hpa grcm-api-hpa
```

### Resource Limits

Per pod resource configuration:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

**Monitoring resource usage:**
```bash
kubectl top pods
kubectl top nodes
```

### Health Checks

**Liveness probe:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

**Readiness probe:**
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

### Rolling Updates

**Update deployment image:**
```bash
kubectl set image deployment/grcm-api \
  grcm-api=ghcr.io/username/grcm:v1.2.0
```

**Monitor rollout:**
```bash
kubectl rollout status deployment/grcm-api
kubectl rollout history deployment/grcm-api
```

**Rollback if needed:**
```bash
kubectl rollout undo deployment/grcm-api
kubectl rollout undo deployment/grcm-api --to-revision=2
```

---

## Configuration

### GRCM Configuration

The main configuration file is `config/grcm_default.yaml`:

```yaml
input_dim: 16
freq_dim: 8
memory_size: 32

attention:
  bandwidth: 1.0
  coherence_threshold: 0.7

phi:
  awareness_threshold: 1.5

qualia:
  low_var_threshold: 0.2
  high_freq_threshold: 0.5

ethical:
  enable: true
  conflict_threshold: 0.6
  halt_on_conflict: true
```

**Override configuration:**
```bash
# Via environment variable
export GRCM_CONFIG_PATH=/path/to/custom_config.yaml

# Via Docker volume mount
docker run -v $(pwd)/config:/app/config grcm:latest

# Via Kubernetes ConfigMap
kubectl create configmap grcm-config --from-file=config/grcm_default.yaml
```

### BentoML Configuration

Create `bentofile.yaml` for custom BentoML builds:

```yaml
service: "service.py:svc"
labels:
  owner: your-team
  project: grcm
include:
  - "grcm/"
  - "config/"
  - "requirements_grcm.txt"
python:
  packages:
    - torch
    - numpy
    - pyyaml
docker:
  distro: debian
  python_version: "3.10"
  system_packages:
    - git
  setup_script: "./setup.sh"
```

---

## Monitoring

### Prometheus Metrics

GRCM exposes the following metrics at `/metrics`:

**Core Metrics:**
- `grcm_phi` - Integrated information (Φ)
- `grcm_coherence_mean` - Average coherence
- `grcm_coherence_std` - Coherence standard deviation
- `grcm_qualia_calm` - Calm qualia probability
- `grcm_qualia_alert` - Alert qualia probability
- `grcm_qualia_curious` - Curious qualia probability
- `grcm_qualia_conflicted` - Conflicted qualia probability

**Performance Metrics:**
- `grcm_inference_latency_ms` - Inference latency histogram
- `grcm_memory_usage_mb` - Memory usage in MB
- `grcm_inference_total` - Total inference count

**API Metrics:**
- `grcm_api_requests_total` - Total API requests
- `grcm_api_errors_total` - Total API errors
- `grcm_ethical_halts_total` - Ethical halt events

### Grafana Dashboards

**Import GRCM dashboard:**
1. Open Grafana at http://localhost:3000
2. Navigate to Dashboards → Import
3. Upload `deployment/grafana-datasources.yml`
4. Select Prometheus datasource

**Dashboard panels:**
- Phi (Φ) trajectory over time
- Coherence distribution
- Qualia state composition
- Inference latency percentiles (p50, p95, p99)
- Desire alignment gauge
- Memory usage gauge
- Throughput counter
- Ethical halt rate

### Alerts

Prometheus alerts are configured in `deployment/prometheus.yml`:

- **HighLatency**: Inference >100ms for 5 minutes
- **LowCoherence**: Coherence <0.5 for 10 minutes
- **HighEthicalHaltRate**: Halt rate >10% for 5 minutes
- **HighErrorRate**: API errors >5% for 5 minutes
- **HighMemoryUsage**: Memory >1.5GB for 5 minutes
- **ServiceDown**: API down for 2 minutes

**Configure alert notifications:**
```yaml
# In prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Configure Slack webhook
- name: 'slack'
  slack_configs:
    - send_resolved: true
      api_url: 'YOUR_SLACK_WEBHOOK_URL'
      channel: '#alerts'
      title: 'GRCM Alert'
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) includes 7 jobs:

1. **Lint**: Code quality checks (black, isort, flake8)
2. **Test**: Unit and integration tests on Python 3.9, 3.10, 3.11
3. **Build**: Docker image build and push to GHCR
4. **Security**: Trivy vulnerability scanning
5. **Deploy-Staging**: Deploy to staging on develop branch
6. **Deploy-Production**: Deploy to production on releases
7. **Performance**: Benchmark tests on PRs

### Required GitHub Secrets

Configure these secrets in GitHub Settings → Secrets:

```bash
GITHUB_TOKEN                # Auto-provided by GitHub
KUBE_CONFIG_STAGING         # Base64-encoded kubeconfig for staging
KUBE_CONFIG_PROD            # Base64-encoded kubeconfig for production
SLACK_WEBHOOK               # Slack notification webhook
CODECOV_TOKEN               # Codecov upload token (optional)
```

**Encode kubeconfig:**
```bash
cat ~/.kube/config | base64 | pbcopy
```

### Deployment Workflow

**Push to develop:**
```bash
git checkout develop
git add .
git commit -m "feat: add new feature"
git push origin develop
```
→ Triggers: lint, test, build, security, deploy-staging

**Create release:**
```bash
git tag v1.2.0
git push origin v1.2.0
```
→ Triggers: lint, test, build, security, deploy-production

### Manual Deployment

**Build and push Docker image:**
```bash
docker build -t ghcr.io/username/grcm:v1.2.0 .
docker push ghcr.io/username/grcm:v1.2.0
```

**Deploy to Kubernetes:**
```bash
kubectl set image deployment/grcm-api \
  grcm-api=ghcr.io/username/grcm:v1.2.0 -n grcm
kubectl rollout status deployment/grcm-api -n grcm
```

---

## Scaling

### Horizontal Scaling

**Auto-scaling with HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: grcm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: grcm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: grcm_inference_latency_ms
        target:
          type: AverageValue
          averageValue: 50
```

### Vertical Scaling

**Increase pod resources:**
```bash
kubectl set resources deployment grcm-api \
  --requests=cpu=1000m,memory=1Gi \
  --limits=cpu=4000m,memory=4Gi
```

### Load Balancing

**NGINX Ingress with load balancing:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grcm-ingress
  annotations:
    nginx.ingress.kubernetes.io/load-balance: "round_robin"
    nginx.ingress.kubernetes.io/upstream-keepalive-requests: "100"
spec:
  rules:
    - host: grcm.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: grcm-api-service
                port:
                  number: 8000
```

---

## Security

### Image Security

**Scan for vulnerabilities:**
```bash
trivy image grcm:latest
```

**Run as non-root user:**
```dockerfile
USER grcm
```

**Read-only root filesystem:**
```yaml
securityContext:
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000
```

### Network Security

**Network policies:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: grcm-api-network-policy
spec:
  podSelector:
    matchLabels:
      app: grcm-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: nginx-ingress
      ports:
        - protocol: TCP
          port: 8000
```

### Secrets Management

**Create Kubernetes secrets:**
```bash
kubectl create secret generic grcm-secrets \
  --from-literal=mlflow-user=admin \
  --from-literal=mlflow-password=secure_password \
  --from-literal=redis-password=redis_password
```

**Use secrets in deployment:**
```yaml
env:
  - name: MLFLOW_PASSWORD
    valueFrom:
      secretKeyRef:
        name: grcm-secrets
        key: mlflow-password
```

### TLS/HTTPS

**Generate self-signed cert (dev only):**
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt
kubectl create secret tls grcm-tls --cert=tls.crt --key=tls.key
```

**Use cert-manager (production):**
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: grcm-tls
spec:
  secretName: grcm-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - grcm.example.com
```

---

## Troubleshooting

### Common Issues

**Issue: Pod not starting**
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

**Issue: High memory usage**
```bash
# Check resource usage
kubectl top pods

# Check memory metrics
curl http://localhost:8000/metrics | grep memory

# Restart pod
kubectl delete pod <pod-name>
```

**Issue: Slow inference**
```bash
# Check latency metrics
curl http://localhost:8000/metrics | grep latency

# Profile with benchmarks
python examples/benchmark_demo.py

# Enable optimization
export GRCM_ENABLE_COMPILE=true
```

**Issue: Service unreachable**
```bash
# Check service endpoints
kubectl get endpoints

# Port forward for debugging
kubectl port-forward service/grcm-api-service 8000:8000

# Check ingress
kubectl describe ingress grcm-ingress
```

### Debug Mode

**Enable debug logging:**
```bash
export GRCM_LOG_LEVEL=DEBUG
```

**Run with profiler:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run GRCM inference
outputs = model(image_emb, audio_emb, action)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Health Checks

**API health check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1234567890,
  "version": "1.0.0"
}
```

**Kubernetes health checks:**
```bash
kubectl get pods
kubectl describe pod <pod-name>
```

---

## Additional Resources

- **Documentation**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: http://localhost:8000/docs (when running)
- **MLflow UI**: http://localhost:5000
- **Grafana Dashboards**: http://localhost:3000
- **GitHub Repository**: https://github.com/yourusername/grcm
- **Issues**: https://github.com/yourusername/grcm/issues

---

## Support

For deployment issues:
1. Check this guide first
2. Review logs and metrics
3. Check GitHub issues
4. Open a new issue with logs and configuration

---

**Last Updated**: 2024
**Version**: 1.0.0
