# Muzzle Deployment Configuration

This directory contains all deployment-related configurations for the Muzzle content moderation API.

## Structure

```
deployment/
├── docker/               # Docker containerization
│   ├── Dockerfile        # Multi-stage container build
│   └── docker-compose.yml # Development environment setup
└── k8s/                  # Kubernetes manifests (planned)
```

## Docker Setup

### Dockerfile

**Multi-stage build optimized for:**
- **Build caching** - Separate dependency and application layers
- **Security** - Non-root user execution
- **Size optimization** - Minimal production image
- **Health checks** - Built-in container health monitoring

**Build stages:**
1. **Base stage** - System dependencies and Python setup
2. **Dependencies stage** - Python package installation
3. **Production stage** - Application code and runtime setup

**Build the image:**
```bash
docker build -f deployment/docker/Dockerfile -t muzzle-api .
```

**Run the container:**
```bash
docker run -p 8000:8000 muzzle-api
```

### Docker Compose

**Complete development environment including:**
- **muzzle-api** - Main FastAPI application
- **postgres** - PostgreSQL database with persistent storage
- **redis** - Redis server for caching and task queues
- **pgadmin** - Database administration interface (optional profile)
- **celery-worker** - Background task processing (optional profile)

**Start development environment:**
```bash
# Basic services (API, database, Redis)
docker-compose -f deployment/docker/docker-compose.yml up

# With optional services (pgAdmin, Celery)
docker-compose -f deployment/docker/docker-compose.yml --profile optional up
```

**Environment variables:**
- Copy `.env.example` to `.env` and configure as needed
- Database and Redis URLs are pre-configured for Docker networking

## Service Configuration

### muzzle-api
- **Port:** 8000 (mapped to host)
- **Health check:** `GET /health` every 30 seconds
- **Restart policy:** Always restart on failure
- **Volume mounts:** Source code for development hot-reload

### postgres
- **Port:** 5432 (internal), 5433 (host mapped)
- **Database:** `muzzle`
- **Credentials:** `postgres/postgres` (development only)
- **Persistent storage:** `postgres_data` volume

### redis
- **Port:** 6379 (internal), 6380 (host mapped)
- **Persistent storage:** `redis_data` volume
- **Configuration:** Basic Redis setup for development

### pgadmin (optional)
- **Port:** 5050 (mapped to host)
- **Access:** http://localhost:5050
- **Credentials:** `admin@muzzle.com / admin`
- **Profile:** `optional` (start with `--profile optional`)

### celery-worker (optional)
- **Background processing** for batch operations
- **Depends on:** Redis and muzzle-api
- **Profile:** `optional` (start with `--profile optional`)

## Common Commands

```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yml up

# Start with optional services
docker-compose -f deployment/docker/docker-compose.yml --profile optional up

# Start in background
docker-compose -f deployment/docker/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f muzzle-api

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down

# Stop and remove volumes (clean slate)
docker-compose -f deployment/docker/docker-compose.yml down -v

# Build and start (after code changes)
docker-compose -f deployment/docker/docker-compose.yml up --build
```

## Production Considerations

**Current setup is optimized for development.** For production deployment:

1. **Environment variables** - Use proper secrets management
2. **Database credentials** - Use strong passwords and restricted access
3. **SSL/TLS** - Add HTTPS termination (nginx, load balancer)
4. **Resource limits** - Set appropriate CPU/memory limits
5. **Monitoring** - Add logging aggregation and metrics collection
6. **Backup strategy** - Implement database backup procedures

## Kubernetes (Planned)

The `k8s/` directory will contain:
- **Deployment manifests** for the API service
- **Service definitions** for internal/external access
- **ConfigMaps** for environment-specific configuration
- **Secrets** for sensitive data management
- **Ingress controllers** for routing and SSL termination
- **Persistent Volume Claims** for database storage

## Troubleshooting

**Common issues:**

1. **Port conflicts** - Ensure ports 8000, 5433, 6380, 5050 are available
2. **Permission errors** - Check Docker daemon permissions
3. **Build failures** - Clear Docker cache: `docker system prune -a`
4. **Database connection** - Verify PostgreSQL is running and accessible
5. **Hot reload issues** - Ensure source code is properly mounted

**Check service health:**
```bash
# API health
curl http://localhost:8000/health

# Database connection
docker-compose -f deployment/docker/docker-compose.yml exec postgres psql -U postgres -d muzzle -c "SELECT 1;"

# Redis connection
docker-compose -f deployment/docker/docker-compose.yml exec redis redis-cli ping
```
