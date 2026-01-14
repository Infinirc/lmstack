# Production Guide

Best practices for running LMStack in production environments.

## Security

### Authentication

1. **Change Default Credentials**: Always change default passwords and API keys
2. **Use Strong Secrets**: Generate cryptographically secure secret keys

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Network Security

1. **Enable HTTPS**: Always use TLS in production

```nginx
server {
    listen 443 ssl;
    server_name lmstack.example.com;

    ssl_certificate /etc/ssl/certs/lmstack.crt;
    ssl_certificate_key /etc/ssl/private/lmstack.key;

    location / {
        proxy_pass http://frontend:80;
    }

    location /api {
        proxy_pass http://backend:8000;
    }
}
```

2. **Firewall Rules**: Restrict access to necessary ports only

```bash
# Allow only HTTPS
ufw allow 443/tcp
ufw deny 8000/tcp  # Block direct backend access
```

3. **CORS Configuration**: Restrict allowed origins

```bash
CORS_ORIGINS=https://lmstack.example.com
```

### API Security

1. **Rate Limiting**: Implement rate limiting to prevent abuse

```python
# Using slowapi
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/endpoint")
@limiter.limit("100/minute")
async def endpoint():
    pass
```

2. **API Key Rotation**: Regularly rotate API keys
3. **Audit Logging**: Log all API access for security auditing

## Performance

### Backend Optimization

1. **Database Connection Pooling**

```python
# SQLAlchemy configuration
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
```

2. **Async Operations**: Use async endpoints for I/O-bound operations

3. **Caching**: Implement Redis caching for frequently accessed data

```python
REDIS_URL=redis://localhost:6379
CACHE_TTL=300  # 5 minutes
```

### Frontend Optimization

1. **Enable Compression**

```nginx
gzip on;
gzip_types text/plain application/json application/javascript text/css;
```

2. **CDN**: Use a CDN for static assets

3. **Browser Caching**

```nginx
location /assets {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

### Worker Optimization

1. **Model Caching**: Pre-download frequently used models

```bash
# Pre-pull models
docker pull vllm/vllm-openai:latest
```

2. **GPU Memory**: Optimize GPU memory allocation

```json
{
  "gpu_memory_utilization": 0.85,
  "max_model_len": 4096
}
```

## High Availability

### Database HA

1. **PostgreSQL Replication**

```yaml
# Primary
services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: master

  postgres-replica:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-primary
```

2. **Regular Backups**

```bash
# Daily backup cron job
0 2 * * * pg_dump -U lmstack lmstack | gzip > /backups/lmstack-$(date +%Y%m%d).sql.gz
```

### Backend HA

1. **Load Balancing**: Run multiple backend instances

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

2. **Health Checks**

```nginx
upstream backend {
    server backend1:8000 max_fails=3 fail_timeout=30s;
    server backend2:8000 max_fails=3 fail_timeout=30s;
}
```

### Worker Redundancy

1. **Multiple Workers**: Deploy workers across multiple GPU nodes
2. **Automatic Failover**: Configure deployments to restart on healthy workers

## Monitoring

### Metrics Collection

1. **Prometheus Integration**

```python
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total HTTP requests')
request_latency = Histogram('http_request_duration_seconds', 'HTTP request latency')
```

2. **Grafana Dashboards**: Create dashboards for:
   - API request rates and latencies
   - GPU utilization
   - Model inference times
   - Error rates

### Alerting

```yaml
# Prometheus alerting rules
groups:
  - name: lmstack
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: High error rate detected

      - alert: WorkerOffline
        expr: worker_status != 1
        for: 2m
        annotations:
          summary: Worker node is offline
```

### Logging

1. **Structured Logging**

```python
import structlog
logger = structlog.get_logger()

logger.info("deployment_started", deployment_id=id, model=model)
```

2. **Log Aggregation**: Use ELK stack or similar for centralized logging

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U lmstack lmstack | gzip > /backups/lmstack_$DATE.sql.gz

# Keep only last 7 days
find /backups -name "lmstack_*.sql.gz" -mtime +7 -delete
```

### Disaster Recovery

1. **Document Recovery Procedures**
2. **Regular Recovery Testing**
3. **Off-site Backup Storage**

## Maintenance

### Updates

```bash
# Rolling update with zero downtime
docker compose pull
docker compose up -d --no-deps backend

# Kubernetes rolling update
kubectl set image deployment/backend backend=lmstack/backend:v1.2.0 -n lmstack
```

### Database Migrations

```bash
# Always backup before migrations
pg_dump -U lmstack lmstack > pre_migration_backup.sql

# Run migrations
alembic upgrade head
```

## Checklist

Before going to production, ensure:

- [ ] HTTPS enabled
- [ ] Strong passwords and secrets configured
- [ ] Database backups automated
- [ ] Monitoring and alerting set up
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] Log aggregation configured
- [ ] Recovery procedures documented and tested
- [ ] Resource limits set for all containers
- [ ] Health checks configured
