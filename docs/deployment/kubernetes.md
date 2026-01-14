# Kubernetes Deployment

Deploy LMStack on Kubernetes for production-grade scalability and high availability.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.x (optional)
- NVIDIA GPU Operator (for GPU workers)
- Persistent storage provisioner

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Ingress   │  │   Backend   │  │  PostgreSQL │     │
│  │  Controller │──│   Service   │──│   StatefulSet│    │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │                               │
│         ▼               ▼                               │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │  Frontend   │  │   Backend   │                      │
│  │ Deployment  │  │ Deployment  │                      │
│  └─────────────┘  └─────────────┘                      │
│                                                          │
│  ┌──────────────────────────────────────────────┐      │
│  │              GPU Node Pool                    │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │      │
│  │  │Worker 1 │  │Worker 2 │  │Worker N │      │      │
│  │  │DaemonSet│  │DaemonSet│  │DaemonSet│      │      │
│  │  └─────────┘  └─────────┘  └─────────┘      │      │
│  └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lmstack
```

```bash
kubectl apply -f namespace.yaml
```

## Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lmstack-secrets
  namespace: lmstack
type: Opaque
stringData:
  database-url: postgresql://lmstack:password@postgres:5432/lmstack
  secret-key: your-secret-key-here
  worker-token: worker-auth-token
```

```bash
kubectl apply -f secrets.yaml
```

## Database

### PostgreSQL StatefulSet

```yaml
# postgres.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: lmstack
spec:
  ports:
    - port: 5432
  selector:
    app: postgres
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: lmstack
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15-alpine
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_USER
              value: lmstack
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: lmstack-secrets
                  key: postgres-password
            - name: POSTGRES_DB
              value: lmstack
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

## Backend

### Deployment

```yaml
# backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: lmstack
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
        - name: backend
          image: lmstack/backend:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: lmstack-secrets
                  key: database-url
            - name: SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: lmstack-secrets
                  key: secret-key
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 2000m
              memory: 2Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: lmstack
spec:
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app: backend
```

## Frontend

### Deployment

```yaml
# frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: lmstack
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: lmstack/frontend:latest
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: lmstack
spec:
  ports:
    - port: 80
      targetPort: 80
  selector:
    app: frontend
```

## Worker

### DaemonSet for GPU Nodes

```yaml
# worker.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: worker
  namespace: lmstack
spec:
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: worker
          image: lmstack/worker:latest
          env:
            - name: BACKEND_URL
              value: http://backend:8088
            - name: WORKER_TOKEN
              valueFrom:
                secretKeyRef:
                  name: lmstack-secrets
                  key: worker-token
            - name: WORKER_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: docker-socket
              mountPath: /var/run/docker.sock
          resources:
            limits:
              nvidia.com/gpu: 1
      volumes:
        - name: docker-socket
          hostPath:
            path: /var/run/docker.sock
```

## Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lmstack
  namespace: lmstack
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "0"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  ingressClassName: nginx
  rules:
    - host: lmstack.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8000
          - path: /v1
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8000
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
  tls:
    - hosts:
        - lmstack.example.com
      secretName: lmstack-tls
```

## Deployment Commands

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres.yaml
kubectl apply -f backend.yaml
kubectl apply -f frontend.yaml
kubectl apply -f worker.yaml
kubectl apply -f ingress.yaml

# Check status
kubectl get pods -n lmstack
kubectl get services -n lmstack
```

## Monitoring

### Pod Status

```bash
kubectl get pods -n lmstack -w
```

### Logs

```bash
kubectl logs -n lmstack -l app=backend -f
kubectl logs -n lmstack -l app=worker -f
```

### Resource Usage

```bash
kubectl top pods -n lmstack
```

## Scaling

```bash
# Scale backend
kubectl scale deployment backend -n lmstack --replicas=4

# HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment backend -n lmstack --min=2 --max=10 --cpu-percent=80
```

## Next Steps

- [Production Guide](production.md) - Security and optimization tips
