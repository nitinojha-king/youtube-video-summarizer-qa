# VidLens: Autonomous Video Intelligence Agent

An end-to-end AI agent that extracts, transcribes, summarizes, and enables grounded conversational question-answering over YouTube and web videos.

VidLens combines high-throughput speech recognition, adaptive hierarchical summarization (Stuffing & Map-Reduce), and dense vector Retrieval-Augmented Generation (RAG) to turn hours of spoken video content into searchable, interactive knowledge in seconds.

---

## Key Features

- **Direct Video Ingestion (`yt-dlp`)**: Streams the optimal audio track directly from any video URL into an isolated temporary workspace without disk bloat.
- **Ultra-Fast Speech Recognition (Whisper Large-v3)**: Transcribes spoken audio into punctuated, cased text in ~10–15 seconds for a 30-minute recording using Groq's LPU inference.
- **Adaptive Dual-Route Summarization**:
  - **Single-Pass Stuffing** ($\le 60,000$ characters): Delivers direct, coherent synthesis for standard videos.
  - **Hierarchical Map-Reduce** ($> 60,000$ characters): Recursively partitions long-form transcripts (lectures, webinars) into 12,000-character semantic chunks, maps key insights, and synthesizes a unified brief under 450 words.
- **Grounded Conversational RAG**:
  - Segments transcripts into 1,000-character passages (150-character overlap).
  - Embeds passages using **NVIDIA Nemotron-3 Embed 1B** into an in-memory **FAISS** vector store.
  - Queries **NVIDIA Nemotron-3.5 Lightning** (with automated fallback to **Nemotron-3 Ultra 550B**).
  - Strictly enforces context grounding: answers are derived exclusively from retrieved excerpts with visible citations, eliminating hallucinations.
- **Editorial Streamlit Interface**: Clean, typography-driven web client featuring embedded video playback, transcript metrics, downloadable briefs (`.md`) and raw transcripts (`.txt`), and interactive chat with source excerpt chips.

---

## Architecture & Workflow

```text
[ Video URL ]
      │
      ▼
[ yt-dlp Audio Ingestion ]  ────────►  Extracts native audio stream (.webm / .m4a)
      │
      ▼
[ Groq Whisper Large-v3 ]   ────────►  Sub-15s transcription with full punctuation
      │
      ├──────────────────────────────────────────┐
      ▼                                          ▼
[ Adaptive Summarizer ]                [ Semantic Vector Indexing ]
  ├─ Stuffing (<= 60k chars)             ├─ Recursive text splitting (1000/150)
  └─ Map-Reduce (> 60k chars)            ├─ NVIDIA Nemotron-3 Embed 1B
      │                                  └─ FAISS Vector Database
      ▼                                          │
[ Structured Brief ]                             ▼
(Overview, Key Points, Takeaways)      [ Grounded RAG QA Engine ]
                                         ├─ Top-4 k-NN Excerpt Retrieval
                                         └─ Nemotron-3.5 Inference + Fallback
                                                 │
                                                 ▼
                                     [ Streamlit Web Application ]
                                     (Chat, Metrics, Source Citations, Exports)
```

---

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Audio Extraction** | `yt-dlp` | Audio demuxing and streaming |
| **Speech-to-Text** | `Groq Cloud` (`whisper-large-v3`) | Hardware-accelerated speech transcription |
| **Orchestration** | `LangChain` | Prompt chaining, Map-Reduce pipelines, output parsers |
| **Embeddings** | `NVIDIA Nemotron-3 Embed 1B` | High-dimensional dense semantic vector representations |
| **Vector Store** | `FAISS` (`faiss-cpu`) | Sub-millisecond in-memory similarity search |
| **Language Model** | `NVIDIA Nemotron-3.5 Lightning` | Primary generative inference via OpenRouter |
| **Fallback LLM** | `NVIDIA Nemotron-3 Ultra 550B` | Automated failover for high availability |
| **Frontend UI** | `Streamlit` | Interactive editorial web application |

---

## Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/nitinojha-king/youtube-video-summarizer-qa.git
cd youtube-video-summarizer-qa
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (or copy `.env.example`):

```bash
cp .env.example .env
```

Add your API keys to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

- Get a **Groq API Key**: [console.groq.com](https://console.groq.com)
- Get an **OpenRouter API Key**: [openrouter.ai](https://openrouter.ai)

---

## Running the Application

### Option A: Interactive Web UI (Recommended)

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`. Paste any YouTube or video URL and click **Analyze video**.

### Option B: Command-Line Interface (CLI)

Run the standalone CLI pipeline:

```bash
python main.py
```

Follow the prompts in the terminal to enter a video URL, view the generated summary, and ask questions interactively.

---

## Project Structure

```text
├── app.py                             # Streamlit web application & UI components
├── main.py                            # Core pipeline: audio, transcription, summarizer, RAG
├── requirements.txt                   # Project Python dependencies
├── .env.example                       # Template for required environment variables
├── .gitignore                         # Git exclusion rules
├── .streamlit/
│   └── config.toml                    # Streamlit styling & theme configuration
├── btech_mini_project_synonims.docx   # Academic project synopsis document
├── BTech_Project_Synopsis_Format.docx # Academic project synopsis format
└── README.md                          # Project documentation
```

---

## License

This project is licensed under the MIT License.
