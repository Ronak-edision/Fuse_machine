# Fuse_week2

[![CCDS-Project%20template-328F97?logo=cookiecutter](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

This project has been updated to follow the principles of a Twelve-Factor App, emphasizing robustness, scalability, and maintainability. It now includes separate backend and frontend components, containerization with Docker, and explicit dependency management.
This project is an image captioning application built with a separated backend and frontend. The backend uses FastAPI to provide an API, and the frontend uses Streamlit for the user interface.  It utilizes Docker for containerization and `docker-compose` for orchestration.

**Project Repository:** [https://github.com/Ronak-edision/Fuse_machine/tree/main/Week2/12_factor_app](https://github.com/Ronak-edision/Fuse_machine/tree/main/Week2/12_factor_app)

## Project Organization

├── data/
│   ├── external/                   # External data (e.g., captions.txt)
│   │   └── captions.txt
│   └── raw/                        # Raw data (e.g., Images/, not in repo)
│       └── Images/                 # Flickr8k validation images (dockerized)
├── models/                         # Trained models
│   ├── BestModel.pth
│   ├── vocab.pkl
│   └── EncodedImageValResNet.pkl
├── src/
│   ├── backend/
│   │   ├── config.py
│   │   ├── init.py
│   │   ├── main.py                 # Backend application entry point
│   │   └── models.py               # Backend data models
│   ├── frontend/
│   │   ├── init.py
│   │   ├── app.py                  # Frontend application entry point (e.g., Flask)
│   │   └── config.py
│   ├── init.py
│   └── config.py                   # Project-wide configurations (if any)
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml         # Pre-commit hooks for code quality
├── docker-compose.yml              # Orchestrates backend and frontend Docker containers
├── Dockerfile.backend              # Docker configuration for the backend
├── Dockerfile.frontend             # Docker configuration for the frontend
├── requirements.backend.txt         # Dependencies for the backend application
├── requirements.frontend.txt        # Dependencies for the frontend application
├── install-extensions.sh           # Script for installing extensions (if applicable)
├── Makefile                        # Convenience commands for development tasks
├── pyproject.toml                  # Project configuration (e.g., build settings)
├── README.md                      
└── requirements.txt        

**Key Components and Technologies:**

* **Backend:**
    * `FastAPI`:  Used to build the backend API (`src/backend/main.py`). The API handles image caption prediction requests.
    * Python
    * PyTorch:  Used for the image captioning model (`src/backend/model.py`).
* **Frontend:**
    * `Streamlit`: Used to create the interactive web application (`src/frontend/app.py`).  It allows users to select images and view generated captions.
    * Python
    * Pillow (PIL):  Used for image processing in the frontend.
    * `requests`:  Used by the frontend to communicate with the backend API.
* **Data Handling:**
    * The project uses a specific directory structure (`data/`) to organize raw images (`data/raw/Images/`), caption data (`data/external/captions.txt`), and processed model files (`models/`).
    * `pandas`: Used in both frontend and backend to handle data, particularly for loading and processing the captions file.
* **Configuration:**
    * Configuration is managed using `config.py` files in both `src/backend` and `src/frontend` and a top-level `src/config.py`.  These files handle paths to data, models, etc.
    * `python-dotenv`: Used for loading environment variables.
    * `loguru`:  Used for logging.
* **Containerization:**
    * `Docker`: Used to containerize both the backend and frontend applications, ensuring consistent environments.  `Dockerfile.backend` and `Dockerfile.frontend` define the build process for each.
    * `docker-compose`:  Used to orchestrate the backend and frontend containers, making it easier to run the entire application.
* **Development Tools:**
    * `Makefile`: Provides convenient commands for common development tasks.
    * `pre-commit`:  Used for setting up pre-commit hooks to enforce code quality.
    * `pyproject.toml`:  Used for project configuration and build system requirements.

This README provides a factual description of the project structure and technologies used, based directly on the code and file organization you provided.