# Muzzle - AI Content Moderation API

This project provides a FastAPI-based microservice that uses machine learning to moderate text and images for toxicity, hate speech, and inappropriate content.

**🚀 NEW: Simplified training with Google Colab!** Train your model with free GPU in ~30 minutes. [See Colab Guide →](COLAB_GUIDE.md)

## Overview

The Muzzle platform is an AI-powered content moderation system that enables automated screening of user-generated content. The system uses a **single comprehensive dataset** (Civil Comments, 300k samples) with **6 core labels** for better training results.

**Key Features:**
- ✅ Single, high-quality dataset (Civil Comments from Google/Jigsaw)
- ✅ 6 comprehensive labels: toxicity, hate speech, harassment, violence, sexual, profanity
- ✅ Google Colab notebook for easy GPU training
- ✅ FastAPI endpoints for production use
- ✅ Pre-trained BERT model ready to deploy

## Structure

```
src/
  muzzle/           # Main Python package
    main.py         # FastAPI application entry point
    api/
      endpoints/    # API route handlers (/moderate/text, /moderate/image)
    core/
      config.py     # Settings and configuration management
    db/
      entities/     # SQLAlchemy database models
      migrations/   # Database migration scripts
    services/       # Business logic for ML inference and moderation
    utils/          # Helper functions and utilities
artifacts/          # Trained ML model files and HuggingFace cache
data/
  raw/              # Original datasets (Kaggle Jigsaw toxic comments)
  processed/        # Cleaned data ready for training
notebooks/          # Jupyter notebooks for model experimentation
scripts/            # Utility scripts for data preprocessing and model training
tests/
  unit/             # Unit tests for individual functions
  integration/      # API endpoint integration tests
deployment/
  docker/           # Docker configuration
    Dockerfile      # Container definition
    docker-compose.yml  # Multi-service development environment
  k8s/              # Kubernetes manifests (planned)
```

## Quick Start

### Option 1: Train on Google Colab (Recommended) 🚀

**Best for:** Getting started quickly with free GPU

1. Push your code to GitHub
2. Open `notebooks/simplified_training.ipynb` in Colab
3. Enable GPU and run all cells
4. Get trained model in ~30-45 minutes

**[📖 Full Colab Guide](COLAB_GUIDE.md)** | **[✅ Quick Checklist](COLAB_CHECKLIST.md)**

### Option 2: Train Locally

**Prerequisites:**
- Python 3.11 or later
- GPU recommended (or CPU with patience)

**Install dependencies:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

**Train the model:**
```bash
# Download dataset
python scripts/data_collection/download_civil_comments.py --sample-size 10000

# Train
python scripts/training/train_bert.py \
  --data-path data/datasets/civil_comments/processed_data.csv \
  --epochs 2 \
  --batch-size 16
```

**Set up environment:**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# At minimum, set DATABASE_URL and REDIS_URL
```

**Run development server:**
```bash
# Start the API server
python3 -m uvicorn muzzle.main:app --reload

# Or using Docker Compose (includes PostgreSQL and Redis)
docker-compose -f deployment/docker/docker-compose.yml up
```

**Access the application:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- API Base: http://localhost:8000/api/

## API Endpoints

**Text Moderation:**
```bash
curl -X POST "http://localhost:8000/api/moderate/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text content here"}'
```

**Image Moderation:**
```bash
curl -X POST "http://localhost:8000/api/moderate/image" \
  -F "image=@path/to/your/image.jpg"
```

## Technology Stack

- **FastAPI** - Modern Python web framework with automatic API documentation
- **Pydantic** - Data validation and settings management
- **SQLAlchemy** - Database ORM with PostgreSQL
- **Redis + Celery** - Async task processing for batch operations
- **HuggingFace Transformers** - Pre-trained ML models (DistilBERT for text)
- **PyTorch** - Deep learning framework for custom models
- **Docker** - Containerization for development and deployment

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python3 -m pytest

# Run linting
black src/ tests/
isort src/ tests/
mypy src/

# Start development server
python3 -m uvicorn muzzle.main:app --reload

# Docker development environment
docker-compose -f deployment/docker/docker-compose.yml up

# Run database migrations (when implemented)
alembic upgrade head
```

## Project Status

**✅ Completed:**
- ✅ Project structure and configuration
- ✅ Simplified training pipeline (single dataset)
- ✅ Google Colab notebook for GPU training
- ✅ 6-label content moderation schema
- ✅ BERT training scripts
- ✅ Data download and preprocessing
- ✅ Basic FastAPI application

**🔄 Next Steps:**
- Integrate trained model into API endpoints
- Add model inference service
- Database for moderation history
- Authentication and rate limiting
- Batch processing with Celery

## Documentation

- **[Colab Training Guide](COLAB_GUIDE.md)** - Train your model with free GPU
- **[Quick Checklist](COLAB_CHECKLIST.md)** - Step-by-step Colab checklist
- **[Label Schema](scripts/preprocessing/label_schema.py)** - 6-label content moderation schema
- **[Training Notebook](notebooks/simplified_training.ipynb)** - Interactive training
- **[Dataset Info](data/datasets/civil_comments/)** - Civil Comments dataset details

## Component Relationships

The main FastAPI application (`main.py`) serves as the entry point and includes routers from the `api/endpoints/` directory. The `core/config.py` manages all application settings loaded from environment variables. The `services/` directory will contain the ML model inference logic, while `db/` handles data persistence. The `deployment/docker/` setup provides a complete development environment with PostgreSQL and Redis.

## Development Approach

This project is being built incrementally with a focus on learning and understanding each component. Each major feature is implemented step-by-step with thorough testing and documentation.

For detailed information about specific components, refer to their respective README files in each directory.