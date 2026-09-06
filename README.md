# Dino Vertex

Dino Vertex is a Flask-based image processing project that performs dinosaur object detection using Detectron2. The app reads image data, runs a custom pre-trained detection model, and returns detected objects along with mask and color information in JSON format.

## Project Overview

This project is designed for a use case where a page or book image is analyzed to identify dinosaur objects in the scene. The backend service exposes a simple API endpoint that accepts request metadata and returns structured detection results.

## Tech Stack

- Python
- Flask
- OpenCV
- PyTorch
- Detectron2
- flask-ngrok

## Project Structure

```text
dino-vertex/
├── dinoapp/
│   ├── app.py
│   └── model/
│       ├── config_2000.yaml
│       └── metadata_2000.pkl
├── .gitignore
├── README.md
└── ...
```

## Main Application

The core logic is in `dinoapp/app.py`.

- Creates a Flask application
- Exposes the `/test` endpoint
- Accepts a JSON payload with image-related data
- Loads the Detectron2 configuration
- Runs the trained model on the supplied image
- Filters out selected classes
- Extracts object masks and dominant mean colors
- Returns results as JSON

## API Endpoint

### `POST /test`

This endpoint expects a JSON body similar to the following:

```json
{
  "book_name": "sample-book",
  "captured_image": "base64-or-image-data",
  "img_url": "https://example.com/image.jpg",
  "user_id": "user-123"
}
```

The app currently validates whether `book_name`, `captured_image`, `img_url`, and `user_id` are present before running the model.

### Response Format

A response contains the metadata and detected object data:

```json
{
  "book_name": "sample-book",
  "user_id": "user-123",
  "page_id": "123",
  "page_iamge_link": "s3//123",
  "objects": {
    "triceratops": "(120, 88, 56)",
    "stegosaurus": "(200, 160, 100)"
  }
}
```

## Model Files

The detection setup depends on the files under `dinoapp/model/`:

- `config_2000.yaml` – Detectron2 configuration for the trained model
- `metadata_2000.pkl` – metadata class mapping used to label predictions

## Setup Instructions

1. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

2. Install the dependencies:

```bash
pip install flask opencv-python torch torchvision detectron2 flask-ngrok
```

3. Ensure the trained model weight file exists and update the path inside `dinoapp/app.py` if needed.

4. Run the application:

```bash
cd dinoapp
python app.py
```

The Flask app starts with `app.run()` and uses `run_with_ngrok(app)` for exposing the app through ngrok when configured.

## Notes

- This project appears to be a prototype / learning project focused on dinosaur detection.
- The model and training weights are not bundled in the repository, so they must be added separately to match the configuration.
- The app is currently structured as a local backend service rather than a full production-ready API.

