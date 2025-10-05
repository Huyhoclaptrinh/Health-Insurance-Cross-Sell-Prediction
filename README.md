# Health Insurance Cross-Sell Prediction

This project is a machine learning pipeline to predict whether a customer who has health insurance would also be interested in purchasing vehicle insurance (cross-selling). The project uses customer data to perform segmentation using clustering and then uses this information to predict the likelihood of a cross-sell.

## Project Structure

```
.
├── app/
├── data/
│   ├── processed/
│   └── raw/
├── health_insurance/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── __pycache__/
│   ├── clustering.py
│   ├── compare_clusters.py
│   ├── data_load.py
│   ├── eda_clustering.py
│   ├── features.py
│   ├── gmm_clustering.py
│   └── visualize_clusters.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── Vehicle Insurance among Health Policyholders Paper.docx
```

-   `src/`: Contains the Python source code for data processing, feature engineering, clustering, and analysis.
-   `data/`: Contains the raw and processed data.
-   `models/`: Stores trained models or preprocessors.
-   `reports/`: Contains analysis reports and visualizations.
-   `Dockerfile` & `docker-compose.yml`: For containerizing and running the application with Docker.
-   `requirements.txt`: A list of the Python dependencies.

## Setup

### Prerequisites

-   Git
-   Python 3.9
-   Docker (optional)

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Health-Insurance-Cross-Sell-Prediction.git
    cd Health-Insurance-Cross-Sell-Prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3.9 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Docker Installation

If you prefer to use Docker, you can build the image using `docker-compose`:

```bash
docker-compose build
```

## How to Run

### 1. Data Preparation

First, you need to process the raw data and create the training, validation, and test splits.

```bash
python src/data_load.py
```

This will take the raw data from `health_insurance/train.csv` and save the processed splits into the `data/processed/` directory.

### 2. Run Clustering Analysis

The main analysis pipeline is defined in the `Dockerfile`. It performs exploratory data analysis on the clusters, compares them, and generates visualizations.

You can run the scripts sequentially:

```bash
python src/eda_clustering.py
python src/compare_clusters.py
python src/visualize_clusters.py
```

### Running with Docker

Alternatively, you can run the entire analysis pipeline using Docker Compose. This will execute the commands specified in the `Dockerfile`.

```bash
docker-compose run --rm web
```

This command will start a container, run the analysis scripts, and then remove the container. The generated reports and figures will be available in the `reports/` directory on your local machine.

## License

This project is licensed under the terms of the LICENSE file.