 # Tap30 Challenge MLOps

An end-to-end MLOps solution for predicting ride demand based on spatiotemporal features. This project demonstrates a complete machine learning pipeline from data ingestion to model deployment with a REST API.

## Original Tutorial Reference

This project is a reimplementation of the following MLOps tutorial:
- **Title**: Tapsi Challenge
- **Creator**: [Alireza Aghamohammadi](https://www.linkedin.com/in/aaghamohammadi/)

## Tech Stack

- **Programming Language**: Python 3.13+
- **Machine Learning**: scikit-learn
- **API Framework**: FastAPI
- **ML Experiment Tracking**: MLflow
- **Containerization**: Docker
- **Container Orchestration**: Kubernetes
- **Package Management**: uv
- **Data Visualization**: Matplotlib
- **Data Storage**: Arvan Cloud S3 (S3-compatible storage)

## Project Structure

```
tap30-challenge-mlops/
├── artifacts/             # Model artifacts and output files
├── config/                # Configuration files
├── data/                  # Data directory
├── logs/                  # Application logs
├── manifests/             # Kubernetes manifests
├── mlruns/                # MLflow experiment tracking data
├── pipeline/              # ML pipeline code
├── src/                   # Core source code
│   ├── data_ingestion.py  # Data loading functionality
│   ├── data_processing.py # Data preprocessing
│   ├── model_training.py  # Model training logic
│   └── logger.py          # Logging functionality
└── web/                   # Web API code
```

## Setup Instructions

### Local Development

1. Ensure you have Python 3.13+ installed
2. Install uv (Python package manager):
   ```
   pip install uv
   ```
3. Clone this repository:
   ```
   git clone <repository-url>
   cd tap30-challenge-mlops
   ```
4. Install dependencies:
   ```
   uv sync
   ```
5. Run the ML pipeline to train the model:
   ```
   python pipeline/run.py
   ```
6. Start the web service:
   ```
   python web/application.py
   ```

### Using Docker

1. Ensure Docker and Docker Compose are installed
2. Build and run the application:
   ```
   docker-compose up --build
   ```
3. The API will be available at http://localhost:8080

### Kubernetes Deployment

1. Build and push the Docker image:
   ```
   docker build -t pouria98sarmasti/tap30-challenge-mlops:0.1.0 .
   docker push pouria98sarmasti/tap30-challenge-mlops:0.1.0
   ```
2. Create the namespace (if not exists):
   ```
   kubectl create namespace tap30
   ```
3. Apply the Kubernetes manifests:
   ```
   kubectl apply -f manifests/
   ```

## API Documentation

### Prediction Endpoint

**URL**: `/predictin`

**Method**: `POST`

**Request Body**:
```json
{
  "hour_of_day": 12,
  "day": 5,
  "row": 3,
  "col": 2
}
```

**Fields**:
- `hour_of_day`: Hour of the day (0-23)
- `day`: Day index
- `row`: Grid row coordinate (0-7)
- `col`: Grid column coordinate (0-7)

**Response Body**:
```json
{
  "demand": 42
}
```

**Fields**:
- `demand`: Predicted demand (integer)

## Example API Usage

```bash
curl -X POST "http://localhost:8080/predictin" \
     -H "Content-Type: application/json" \
     -d '{
           "hour_of_day": 18,
           "day": 2,
           "row": 4,
           "col": 3
         }'
```

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.

```
Copyright 2025 Pouria Sarmasti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```