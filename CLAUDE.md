# ARCHINZA SEARCH PIPELINE
## Architecture Overview

**Scale**: 5,000-10,000 images | <500 queries/day
**Target Response**: 2-4 seconds (CPU-optimized)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHINZA SEARCH PIPELINE                      │
│              (5-10K images | <500 queries/day)                   │
└─────────────────────────────────────────────────────────────────┘

User Input (Image/Text/Both)
            │
            ▼
    ┌───────────────┐
    │  FastAPI      │  ──────────────────────────┐
    │  (Direct)     │                            │
    └───────┬───────┘                            │
            │                                    │
            ▼                                    ▼
    ┌───────────────┐                   ┌───────────────┐
    │ User Context  │                   │  Input Router │
    │   Detection   │                   │ (Case 1/2/3)  │
    └───────┬───────┘                   └───────┬───────┘
            │                                    │
            └────────────┬───────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   CLIP Model        │  ← Single model for ALL:
              │   (CPU + ONNX)      │    • Image embeddings
              └─────────┬───────────┘    • Text embeddings
                        │                • Label detection
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
      ┌─────────┐ ┌─────────┐ ┌─────────┐
      │ Labels  │ │ Image   │ │ Text    │
      │ (CLIP)  │ │ Embed   │ │ Embed   │
      └────┬────┘ └────┬────┘ └────┬────┘
           │           │           │
           │           └─────┬─────┘
           │                 │
           │                 ▼
           │        ┌─────────────────┐
           │        │  Fusion Layer   │  ← Weighted average
           │        │  (if both)      │
           │        └────────┬────────┘
           │                 │
           │                 ▼
           │        ┌─────────────────┐
           │        │   FAISS Index   │  ← In-memory vector search
           │        │   (< 20ms)      │    ~10MB for 10K images
           │        └────────┬────────┘
           │                 │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  S3 Image URLs  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   GPT-4o-mini   │  ← Fast, cheap LLM
           │   (Async call)  │    • Summary
           └────────┬────────┘    • Follow-up (user-type aware)
                    │
                    ▼
           ┌─────────────────┐
           │  JSON Response  │
           └─────────────────┘
```

---

## Latency Breakdown (CPU-Optimized)

| Step | Component | Time | Optimization |
|------|-----------|------|--------------|
| 1 | FastAPI | ~10ms | Direct, no API Gateway |
| 2 | CLIP Embedding | ~600-1000ms | ONNX runtime, CPU optimized |
| 3 | CLIP Labels | ~200-400ms | Batch with embedding |
| 4 | FAISS Search | ~10-20ms | In-memory, only 10K vectors |
| 5 | S3 URL Generation | ~50ms | Pre-signed URLs |
| 6 | GPT-4o-mini | ~800-1500ms | Concise prompt |
| **Total** | | **~1.7-3.5s** | ✅ Acceptable for scale |

**Note**: With ONNX optimization and model caching, CPU inference is practical for <500 queries/day.

---

## Tech Stack Summary

| Component | Technology | Cost |
|-----------|------------|------|
| Embeddings | CLIP (ViT-B/32) + ONNX | Free |
| Vector DB | FAISS (in-memory) | Free |
| Labels | CLIP Zero-Shot | Free |
| LLM | GPT-4o-mini | ~$0.00015/query |
| Storage | S3 | Minimal |
| API | FastAPI on EC2 | ~$30-60/month |

**Why no Redis?** At <500 queries/day, in-memory FAISS is sufficient. No caching layer needed.

---

## Infrastructure (Low-Cost)

```
┌─────────────────────────────────────────┐
│     DEPLOYMENT STACK (Small Scale)      │
├─────────────────────────────────────────┤
│  API Layer:     FastAPI (direct)        │
│  Compute:       t3.large (CPU)          │
│  Vector Index:  FAISS (~10MB in RAM)    │
│  Storage:       S3                      │
│  LLM:           OpenAI API              │
└─────────────────────────────────────────┘

Estimated Monthly Cost:
- EC2 t3.large:    ~$60/month
- S3 (10K images): ~$5/month
- OpenAI (500 queries): ~$2.50/month
─────────────────────────────
Total:             ~$67.50/month
```

### Instance Options

| Instance | vCPU | RAM | Cost/month | Use Case |
|----------|------|-----|------------|----------|
| t3.medium | 2 | 4GB | ~$30 | Dev/testing |
| t3.large | 2 | 8GB | ~$60 | Production (recommended) |
| t3.xlarge | 4 | 16GB | ~$120 | Faster inference |

### Why t3.large?
- **8GB RAM**: Enough for CLIP model (~400MB) + FAISS index (~10MB) + overhead
- **Burstable CPU**: Good for sporadic traffic (<500/day)
- **Cost-effective**: 10x cheaper than GPU instance

### Scaling Path
When you exceed 500 queries/day or need <2s response:
1. **First**: Upgrade to t3.xlarge (~$120/month)
2. **Then**: Add ONNX quantization for faster CPU inference
3. **Finally**: Switch to g4dn.xlarge GPU (~$380/month) for high traffic

---

## Three Input Cases

### Case 1: Image Only
```
Image → CLIP Embed → CLIP Labels → FAISS Search → S3 URLs → GPT Summary → Response
```

### Case 2: Text Only
```
Text → CLIP Embed → FAISS Search → S3 URLs → GPT Summary → Response
```

### Case 3: Image + Text
```
Image + Text → CLIP Embed (both) → Fusion → FAISS Search → S3 URLs → GPT Summary → Response
```

---

## Cost Comparison

| Stack | Monthly Cost | Notes |
|-------|-------------|-------|
| **Original AWS Stack** | $800-1500 | Titan + OpenSearch + Rekognition |
| **GPU Optimized** | ~$435 | g4dn.xlarge + Redis |
| **Your Scale (CPU)** | **~$67.50** | t3.large, no Redis |

### Savings: ~90% vs original AWS stack

---

## FAISS Index Size

For 10,000 images with 512-dimensional embeddings:
- **Vector data**: 10,000 × 512 × 4 bytes = ~20MB
- **With metadata**: ~25-30MB total
- **RAM usage**: Easily fits in t3.large's 8GB

---

## Performance Optimizations for CPU

1. **ONNX Runtime** - Convert CLIP to ONNX for 2-3x faster CPU inference
2. **Model Quantization** - INT8 quantization for smaller/faster model
3. **Batch Labels** - Get labels in same forward pass as embedding
4. **Pre-warm Model** - Load model on startup, keep in memory
5. **Async LLM** - Don't block on GPT-4o-mini response

---

## Quick Start Commands

```bash
# Install dependencies
pip install transformers torch faiss-cpu fastapi uvicorn openai boto3 pillow onnxruntime

# Run locally
uvicorn app.main:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/search \
  -F "image=@test.jpg" \
  -F "user_type=professional"
```