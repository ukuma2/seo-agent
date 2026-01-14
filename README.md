# Advanced SEO Content Evaluator

A Python-based AI tool that crawls a webpage, performs real-time keyword research using DuckDuckGo, and uses Google Gemini to provide a comprehensive SEO scoring and optimization audit.

## Features
- **URL Crawling**: Extracts main content, title, and metadata.
- **Live Research**: Finds trending keywords and common questions via DuckDuckGo.
- **AI Analysis**: Gemini (Flash) analyzes the content + research to score SEO and suggest specific rewrites (Title, Meta, Intro).
- **Streamlit UI**: Clean, interactive dashboard.

## Installation & Local Run

1. **Clone the repo** (or download files):
   ```bash
   git clone <your-repo-url>
   cd seo-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Key**:
   - Create a file named `secrets.toml` inside a `.streamlit` folder.
   - Path: `.streamlit/secrets.toml`
   - Content:
     ```toml
     GOOGLE_API_KEY = "your-google-gemini-api-key"
     ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud (Free)

This app is "GitHub-ready". To deploy for free:

1. Push this code to a **GitHub repository** (Public or Private).
   - *Note: Do NOT commit your `.streamlit/secrets.toml` file. The `.gitignore` file included here prevents this.*

2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select your new repository.
4. Click **Deploy**.
5. **Important**: You must add your API Key in the Streamlit Cloud Dashboard:
   - Go to your App Settings -> **Secrets**.
   - Add:
     ```toml
     GOOGLE_API_KEY = "your-google-gemini-api-key"
     ```
6. Save, and your app will be live globally!

## Tech Stack
- **Streamlit**: Frontend
- **LangChain**: LLM Orchestration
- **Google Gemini**: AI Model
- **DuckDuckGo**: Search Tool
