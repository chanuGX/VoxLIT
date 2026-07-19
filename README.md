# VoxLIT — Learning Interpretability Tool for Voice Models

<p align="center">
  <a href="https://github.com/chanuGX/VoxLIT">
    <img src="https://img.shields.io/badge/version-v1.0-blue" alt="Version"/>
  </a>
  <a href="https://github.com/chanuGX/VoxLIT/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/chanuGX/VoxLIT" alt="License"/>
  </a>
  <a href="https://github.com/chanuGX/VoxLIT/stargazers">
    <img src="https://img.shields.io/github/stars/chanuGX/VoxLIT" alt="Stars"/>
  </a>
  <a href="https://github.com/chanuGX/VoxLIT/issues">
    <img src="https://img.shields.io/github/issues/chanuGX/VoxLIT" alt="Issues"/>
  </a>
</p>

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram).

VoxLIT extends the interpretability paradigm to audio models, providing researchers and developers with tools to analyze and debug speech models with greater transparency. Through interactive visualizations, attention mechanisms, and perturbation analyses, you can gain deeper insights into how your audio models make decisions.

## Features

- **Audio Data Management**: Upload and manage audio datasets with metadata
- **Waveform Visualization**: Interactive waveform viewer with playback controls
- **Model Prediction Analysis**: Examine model predictions and confidence scores
- **Attention Visualization**: Explore attention patterns in transformer-based audio models
- **Embedding Analysis**: Visualize high-dimensional audio embeddings in 2D/3D space
- **Saliency Mapping**: Identify important regions in audio input using gradient-based methods
- **Perturbation Tools**: Apply various audio perturbations to test model robustness
- **Interactive Dashboard**: Comprehensive interface for exploring model behavior
- **Faithful Emotion Recognition**: Loads the wav2vec2 emotion model with its trained classifier head via a custom model class (the stock `Wav2Vec2ForSequenceClassification` loader silently re-initializes the head, producing random predictions)

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **UI Framework**: Tailwind CSS + shadcn/ui components
- **State Management**: TanStack Query
- **Data Visualization**: Custom React components with Chart.js integration
- **Audio Processing**: Web Audio API
- **Backend**: FastAPI + Python 3.11
- **Models**: Transformer-based audio models (Whisper, Wav2Vec2)
- **Storage**: Redis for caching predictions and results

## Prerequisites

- **Frontend**:
  - Node.js (v18 or higher)
  - npm or bun package manager

- **Backend**:
  - Python 3.11
  - Docker Desktop (for Redis)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/chanuGX/VoxLIT.git
cd VoxLIT
```

### 2. Set up the Frontend

```bash
cd Frontend
npm install
npm run dev
```

The frontend dev server runs at http://localhost:8080.

### 3. Start Redis in Docker

```bash
# In a new terminal (Docker Desktop must be running)
cd Backend
docker compose up -d
```

The backend defaults to `redis://localhost:6379/0`, so no `.env` is needed for local development. To override settings (e.g., a remote Redis), create `Backend/.env` with `REDIS_URL=...`.

### 4. Set up the Backend

> **Important:** always start uvicorn **from the `Backend/` directory** — the app resolves `data/` and `uploads/` relative to the working directory.

```bash
cd Backend
py -3.11 -m venv .venv          # or: python3.11 -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Unix / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at http://localhost:8000.

Model weights (Whisper, Wav2Vec2) are downloaded from Hugging Face on first startup and cached under `~/.cache/huggingface`. The first launch takes a few minutes; `whisper-large-v3` (~3 GB) is only downloaded if selected in the UI.

### 5. Access the Application

Open your browser and navigate to [http://localhost:8080](http://localhost:8080)

## Project Structure

```
VoxLIT/
├── Frontend/                # React frontend application
│   ├── components/          # React components
│   │   ├── analysis/        # Analysis and perturbation tools
│   │   ├── audio/           # Audio visualization components
│   │   ├── layout/          # Layout components
│   │   ├── panels/          # Dashboard panels
│   │   ├── ui/              # Reusable UI components
│   │   └── visualization/   # Data visualization components
│   ├── hooks/               # Custom React hooks
│   ├── lib/                 # Utility functions
│   └── pages/               # Page components
│
├── Backend/                 # FastAPI backend application
│   ├── app/                 # Application code
│   │   ├── api/             # API routes and endpoints
│   │   ├── core/            # Core functionality
│   │   └── services/        # Business logic services
│   ├── data/                # Sample datasets (git-ignored; add your own)
│   ├── tests/               # Backend tests
│   └── uploads/             # User-uploaded audio files
│
├── CODE_OF_CONDUCT.md       # Community guidelines
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── SECURITY.md              # Security policy
```

## Available Scripts

### Frontend

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

### Backend

- `pytest` - Run backend tests
- `uvicorn app.main:app --reload` - Start the API server in development mode

## Usage

1. **Upload Audio Data**: Use the audio uploader to load your audio files
2. **Select Models**: Choose from available audio models for analysis
3. **Explore Visualizations**:
   - Examine waveforms and spectrograms
   - View model predictions and confidence scores
   - Explore attention patterns and embedding spaces
   - Generate saliency maps to highlight important audio regions
4. **Apply Perturbations**: Test model robustness with various audio perturbations
5. **Analyze Results**: Use the interactive dashboard to gain insights

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Security

For security-related issues, please refer to our [Security Policy](SECURITY.md).

## Acknowledgments

- Inspired by Google's [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)
- Built with modern React ecosystem and TypeScript
- Special thanks to the open-source community for the amazing tools and libraries

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built for audio model interpretability</sub>
</p>
