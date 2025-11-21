# Quick Start Guide

## Testing Your Improvements

### 1. Run All Tests
```bash
# Run complete test suite
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app --cov-report=html
```

### 2. Run Specific Tests
```bash
# API tests only
pytest tests/test_api.py

# Service tests only
pytest tests/test_services.py

# History service integration test
python test_history.py
```

### 3. Start the API Server
```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Test the API

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

#### Text Search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -F "text_query=modern kitchen" \
  -F "user_type=general"
```

#### Image Search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -F "image=@path/to/image.jpg" \
  -F "user_type=professional"
```

#### Combined Search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -F "image=@path/to/image.jpg" \
  -F "text_query=minimalist bedroom" \
  -F "user_type=student"
```

### 5. Test Rate Limiting
```bash
# This should succeed (first 20 requests)
for i in {1..20}; do
  curl -X POST http://localhost:8000/api/v1/search \
    -F "text_query=test"
done

# This should return 429 (21st request)
curl -X POST http://localhost:8000/api/v1/search \
  -F "text_query=test"
```

### 6. View API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Re-indexing Images (With Object Detection)

```bash
# Reset index and re-index all images
python scripts/index_images.py --reset-index --prefix "images/"

# Index specific images
python scripts/index_images.py --s3-keys images/img1.jpg images/img2.jpg

# Index with custom batch size
python scripts/index_images.py --prefix "images/" --batch-size 16
```

---

## Performance Testing

### Measure Latency
```bash
# Install Apache Bench (if not already installed)
# apt-get install apache2-utils  # Linux
# brew install httpd              # Mac

# Test search endpoint (100 requests, 10 concurrent)
ab -n 100 -c 10 -p search.json -T "multipart/form-data; boundary=----" \
  http://localhost:8000/api/v1/search
```

### Monitor Logs
```bash
# Start server with detailed logging
uvicorn app.main:app --reload --log-level debug
```

---

## Verify Improvements

### 1. Check Model Warmup
Look for this in startup logs:
```
INFO - Loading CLIP model...
INFO - Warming up CLIP model with dummy inference...
INFO - Model warmup complete
```

### 2. Check Parallel URL Generation
Search endpoint should use async URL generation:
```python
# In routes.py line 169
urls = await s3_service.get_presigned_urls_async(s3_keys)
```

### 3. Check Rate Limiting
Make 21 requests in quick succession - the 21st should return:
```json
{
  "error": "Rate limit exceeded",
  "detail": "20 per 1 minute"
}
```

### 4. Check Input Validation
```bash
# This should return 400 error
curl -X POST http://localhost:8000/api/v1/search

# This should return 400 (text too long)
curl -X POST http://localhost:8000/api/v1/search \
  -F "text_query=$(python -c 'print("a"*501)')"
```

---

## Environment Setup Checklist

- [ ] Copied `.env.example` to `.env`
- [ ] Updated `.env` with your credentials
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Verified `.env` is in `.gitignore`
- [ ] Rotated any exposed credentials
- [ ] Re-indexed images with object detection

---

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/CLIP-search-engine

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Model Loading Issues
```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/

# Re-download model
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### FAISS Index Issues
```bash
# Reset and rebuild index
python scripts/index_images.py --reset-index --prefix "images/"
```

---

## Performance Benchmarks

### Expected Response Times (CPU)

| Endpoint | Before | After | Improvement |
|----------|--------|-------|-------------|
| First search (cold) | ~2000ms | ~1500ms | 25% faster |
| Text search | ~1800ms | ~1400ms | 22% faster |
| Image search | ~2000ms | ~1600ms | 20% faster |
| URL generation (10) | ~250ms | ~80ms | 68% faster |

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/search` | 20/minute per IP |
| Other endpoints | No limit (add as needed) |

---

## Next Steps

1. **Security:** Rotate exposed credentials immediately
2. **Testing:** Run `pytest` to verify all tests pass
3. **Indexing:** Re-run indexing script to add object detection
4. **Monitoring:** Set up logging/monitoring in production
5. **Optimization:** Consider ONNX conversion if needed

---

## Getting Help

- Check `IMPROVEMENTS.md` for detailed change documentation
- Review API docs at http://localhost:8000/docs
- Check logs for detailed error messages
- Run tests to verify functionality: `pytest -v`
