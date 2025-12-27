# Multi-stage Dockerfile for GRCM production deployment
# Optimized for size and security

# Stage 1: Builder
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_grcm.txt .
COPY setup.py .
COPY README_GRCM.md .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements_grcm.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 grcm && \
    mkdir -p /app && \
    chown -R grcm:grcm /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=grcm:grcm grcm/ ./grcm/
COPY --chown=grcm:grcm config/ ./config/
COPY --chown=grcm:grcm examples/ ./examples/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/mlruns && \
    chown -R grcm:grcm /app/models /app/logs /app/mlruns

# Switch to non-root user
USER grcm

# Expose ports
# 8000: BentoML API
# 7860: Gradio UI
# 5000: MLflow UI
EXPOSE 8000 7860 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grcm; print('healthy')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "grcm.ui"]
