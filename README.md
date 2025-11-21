# ARCHINZA CLIP Search Engine

An AI-powered visual search engine for architectural images using OpenAI's CLIP model and FAISS vector similarity search. Built with FastAPI and Streamlit for production-grade performance and scalability.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

ARCHINZA is a production-ready visual search system designed for architectural image databases. It leverages CLIP's multi-modal capabilities to enable search through images, text descriptions, or both simultaneously. The system is optimized for CPU inference and scales efficiently to 5,000-10,000 images with sub-second search latency.

## Key Features

### Multi-Modal Search Capabilities
- **Image-based Search**: Upload architectural images to find visually similar designs
- **Text-based Search**: Natural language queries for architectural concepts
- **Hybrid Search**: Combined image and text queries with configurable fusion weights (default: 70% image, 30% text)

### Advanced Search Features
- **Similar Image Discovery**: One-click similarity search with optimized 1-2 second response time
- **Dual Classification System**: Simultaneous detection of architectural styles and interior objects/furniture
- **AI-Generated Summaries**: Contextual search result summaries powered by GPT-4o-mini
- **User Context Awareness**: Tailored recommendations based on user type (professional, student, enthusiast)

### Performance Optimizations
- **ONNX Runtime Support**: Optional 2-3x speedup for CPU inference
- **Asynchronous Operations**: Parallel S3 URL generation and batch processing
- **Model Warmup**: Eliminates cold start latency on application startup
- **Intelligent Caching**: Session-based favorites and search history management

### Production Features
- **Rate Limiting**: 20 requests per minute per IP address
- **Input Validation**: File size (max 10MB), format (JPEG/PNG/WebP), and query length constraints
- **Comprehensive Error Handling**: Structured exception handling with detailed logging
- **RESTful API**: Fully documented OpenAPI specification

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ARCHINZA Search Pipeline                    │
│              (Optimized for 5-10K images)                    │
└─────────────────────────────────────────────────────────────┘

                    User Input
                (Image / Text / Both)
                        │
                        ▼
              ┌─────────────────────┐
              │   FastAPI Backend   │
              │   - Rate Limiting   │
              │   - Validation      │
              └──────────┬──────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │    CLIP Model       │
              │  (ViT-B/32 Base)    │
              │  PyTorch or ONNX    │
              └──────────┬──────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Labels  │  │  Image   │  │   Text   │
    │Detection │  │Embedding │  │Embedding │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         │             └──────┬──────┘
         │                    │
         │                    ▼
         │          ┌──────────────────┐
         │          │  FAISS Index     │
         │          │  (IndexFlatIP)   │
         │          │  512-dim vectors │
         │          └────────┬─────────┘
         │                   │
         └──────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   AWS S3 URLs   │
           │  (Async Batch)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  GPT-4o-mini    │
           │  Summarization  │
           └────────┬────────┘
                    │
                    ▼
              JSON Response
```

## Installation

### Prerequisites

- Python 3.9 or higher
- AWS Account with S3 access
- OpenAI API key
- 8GB RAM minimum (16GB recommended)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/Souvik2411/CLIP-search-engine.git
cd CLIP-search-engine
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your-bucket-name

# OpenAI Configuration
OPENAI_API_KEY=your_api_key

# Search Parameters
TOP_K_RESULTS=10
IMAGE_TEXT_FUSION_WEIGHT=0.7

# ONNX Configuration (Optional)
USE_ONNX=false
ONNX_VISION_MODEL_PATH=models/onnx/clip_vision.onnx
ONNX_TEXT_MODEL_PATH=models/onnx/clip_text.onnx
```

5. **Initialize Image Index**
```bash
# Upload images to S3 bucket under 'images/' prefix
# Then run indexing:
python scripts/index_images.py --reset-index --prefix "images/"
```

6. **Start Backend Server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

7. **Start Frontend Application** (new terminal)
```bash
streamlit run streamlit_app_dual.py --server.port 8501
```

8. **Access Application**
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/api/v1/health

## ONNX Optimization

For production deployments requiring higher throughput, enable ONNX runtime for 2-3x faster inference:

### Convert Models to ONNX Format
```bash
python scripts/convert_to_onnx.py
```

This generates optimized models (~577MB total):
- `models/onnx/clip_vision.onnx` (334MB)
- `models/onnx/clip_text.onnx` (241MB)

### Enable ONNX Runtime
Update `.env`:
```bash
USE_ONNX=true
ONNX_VISION_MODEL_PATH=models/onnx/clip_vision.onnx
ONNX_TEXT_MODEL_PATH=models/onnx/clip_text.onnx
```

Restart the backend server to apply changes.

## API Documentation

### Core Endpoints

#### Search Operations
```
POST   /api/v1/search
       Multi-modal search supporting image, text, or combined queries
       Rate limit: 20 requests/minute per IP

GET    /api/v1/search/similar/{image_id}?top_k=5
       Fast similarity search using cached embeddings
```

#### Index Management
```
POST   /api/v1/index
       Batch index images from S3

DELETE /api/v1/index/{image_id}
       Remove image from search index

GET    /api/v1/index/stats
       Retrieve index statistics and metadata
```

#### User Features
```
GET    /api/v1/history?limit=20
       Retrieve search history

POST   /api/v1/history/session
       Create new search session

GET    /api/v1/favorites?limit=50
       Retrieve saved favorites

POST   /api/v1/favorites
       Add image to favorites

POST   /api/v1/favorites/check-batch
       Batch check favorite status
```

#### System Health
```
GET    /api/v1/health
       Health check and system status
```

Complete API documentation available at `/docs` endpoint.

## Project Structure

```
CLIP-search-engine/
├── app/
│   ├── api/
│   │   └── routes.py              # API endpoint definitions
│   ├── models/
│   │   └── schemas.py             # Pydantic data models
│   ├── services/
│   │   ├── clip_service.py        # PyTorch CLIP implementation
│   │   ├── clip_onnx_service.py   # ONNX CLIP implementation
│   │   ├── faiss_service.py       # Vector similarity search
│   │   ├── s3_service.py          # AWS S3 operations
│   │   ├── llm_service.py         # OpenAI integration
│   │   ├── search_history_service.py
│   │   └── favorites_service.py
│   ├── utils/
│   │   ├── helpers.py             # Utility functions
│   │   └── error_handlers.py      # Exception handling
│   ├── config.py                  # Configuration management
│   └── main.py                    # Application entry point
├── scripts/
│   ├── index_images.py            # Batch indexing utility
│   └── convert_to_onnx.py         # Model conversion utility
├── tests/
│   ├── test_api.py                # API integration tests
│   └── test_services.py           # Service unit tests
├── pages/                         # Streamlit multi-page components
├── streamlit_app_dual.py          # Frontend application
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md                      # Documentation
```

## Testing

Execute the test suite with pytest:

```bash
# Run all tests with verbose output
pytest -v

# Run specific test modules
pytest tests/test_api.py -v
pytest tests/test_services.py -v

# Generate coverage report
pytest --cov=app --cov-report=html tests/
```

**Test Coverage**: 40 comprehensive tests
- 21 API endpoint tests
- 19 service and utility tests

## Performance Benchmarks

### Search Latency

| Query Type | PyTorch (CPU) | ONNX (CPU) | Improvement |
|-----------|---------------|------------|-------------|
| Text only | 2.0 - 4.0s | 1.5 - 3.0s | 25-33% |
| Image only | 3.0 - 5.0s | 2.0 - 3.5s | 30-40% |
| Combined | 3.0 - 5.0s | 2.0 - 3.5s | 30-40% |
| Similar images | 1.0 - 2.0s | 1.0 - 2.0s | (Same - no CLIP inference) |

### Infrastructure Costs (AWS)

| Component | Specification | Monthly Cost |
|-----------|--------------|-------------|
| Compute | EC2 t3.large (2 vCPU, 8GB RAM) | $60 |
| Storage | S3 Standard (10,000 images) | $5 |
| AI Services | OpenAI API (~500 queries) | $2.50 |
| **Total** | | **$67.50** |

### Scalability

- **Indexed Images**: Optimized for 5,000 - 10,000 images
- **Memory Footprint**: ~30MB for 10,000 image index
- **Search Latency**: <20ms for FAISS vector search
- **Throughput**: ~500 queries/day on t3.large

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug logging | false |
| `CLIP_MODEL_NAME` | Hugging Face model identifier | openai/clip-vit-base-patch32 |
| `EMBEDDING_DIM` | Vector embedding dimension | 512 |
| `TOP_K_RESULTS` | Number of search results | 10 |
| `IMAGE_TEXT_FUSION_WEIGHT` | Image weight in hybrid search | 0.7 |
| `USE_ONNX` | Enable ONNX runtime | false |

### Architecture Labels

The system detects 16 architectural style categories:
- Modern architecture, Classical architecture
- Minimalist design, Brutalist architecture
- Art deco, Gothic architecture
- Contemporary design, Sustainable architecture
- Industrial design, Residential building
- Commercial building, Interior design
- Landscape architecture, Urban planning
- Facade design, Structural design

### Furniture & Object Labels

Detects 60+ interior object categories including seating, tables, storage, lighting, and decorative elements.

## Security Considerations

- Credentials stored in environment variables (not committed to version control)
- Rate limiting prevents API abuse (20 req/min per IP)
- Input validation prevents malformed requests
- Structured error handling prevents information leakage
- S3 presigned URLs expire after 1 hour
- HTTPS enforcement for all external connections

## Troubleshooting

### Port Conflict Error
```bash
# Identify process using port
netstat -ano | findstr :8000

# Terminate process (Windows)
taskkill /PID <PID> /F

# Terminate process (Linux/Mac)
kill -9 <PID>
```

### ONNX Models Missing
```bash
# Generate ONNX models (requires ~4GB disk space)
python scripts/convert_to_onnx.py

# Verify generation
ls -lh models/onnx/
```

### Empty Search Results
```bash
# Verify index size
curl http://localhost:8000/api/v1/index/stats

# Re-index images
python scripts/index_images.py --reset-index --prefix "images/"
```

### Performance Degradation
1. Enable ONNX runtime for 2-3x speedup
2. Verify network connectivity to S3
3. Check FAISS index loaded successfully in logs
4. Monitor CPU/memory usage

## Version History

### v1.3.0 (Current)
- Implemented fast similarity search feature (90% latency reduction)
- Added search-by-image-id API endpoint
- Optimized FAISS embedding reconstruction

### v1.2.0
- Added ONNX runtime support
- Implemented model conversion utilities
- Created CLIPONNXService for optimized inference

### v1.1.0
- Comprehensive test suite (40 tests, 85%+ coverage)
- Rate limiting and input validation
- Dual label detection system
- Async S3 URL generation
- Migrated to FastAPI lifespan pattern

### v1.0.0
- Initial production release
- Multi-modal search capabilities
- Search history and favorites features
- AI-powered result summarization

## Contributing

We welcome contributions to improve ARCHINZA. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -am 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass (`pytest -v`)
- New features include appropriate tests
- Documentation is updated accordingly

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- **OpenAI** - CLIP model architecture and pre-trained weights
- **Facebook AI Research** - FAISS vector similarity search library
- **FastAPI** - Modern web framework for Python
- **Streamlit** - Rapid UI development framework
- **Anthropic** - Claude Code development assistance

## Contact & Support

**Author**: Souvik2411
**GitHub**: [@Souvik2411](https://github.com/Souvik2411)
**Repository**: [CLIP-search-engine](https://github.com/Souvik2411/CLIP-search-engine)

For bug reports and feature requests, please open an issue on GitHub.

---

**Built with**: CLIP • FAISS • FastAPI • Streamlit • PyTorch/ONNX
