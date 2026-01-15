"""
==============================================================================
ADVANCED SEO CONTENT EVALUATOR
==============================================================================
A robust, error-proof Streamlit application that:
1. Crawls a URL to extract page content.
2. Performs live keyword research using DuckDuckGo (native HTTP, no buggy libs).
3. Analyzes the content with Google Gemini for SEO improvements.

Author: AI Assistant
Version: 2.0 (Error-Proof Refactor)
==============================================================================
"""

import streamlit as st
import json
import requests
import re
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Advanced SEO Evaluator AI",
    page_icon="🚀",
    layout="wide"
)

# System prompt for Gemini - embedded as per requirements
# Enhanced for richer, more detailed output with EXPLANATIONS
GEMINI_SYSTEM_PROMPT = """
You are an elite SEO specialist using modern ranking factors (EEAT, Helpfulness, Core Web Vitals).
You will receive page content and live keyword ideas from search results.

IMPORTANT: Assess the page ID/INTENT before scoring.
- **Navigational/Brand** (e.g., Apple, Google): Score on UX, clarity, and trust. Do NOT penalize for low text density.
- **Informational** (e.g., Blogs, Guides): Score on content depth, keyword coverage, and entity richness.
- **Transactional** (e.g., Product Pages): Score on trust signals, clear CTAs, and performance.

Your task:
1. Determine the Page Intent.
2. Pick the best Primary Keyword based on research context.
3. Analyze the page based on its Intent (not generic rules).
4. Provide exact replacements for Title, Meta, H1, etc., WITH REASONING.

You MUST output valid JSON matching this exact schema (no extra text):
{
    "page_intent": "<string: 'Navigational', 'Informational', 'Transactional'>",
    "seo_score": <integer 0-100>,
    "score_breakdown": {
        "keyword_optimization": <integer 0-25>,
        "content_quality": <integer 0-25>,
        "technical_seo": <integer 0-25>,
        "user_experience": <integer 0-25>
    },
    "critical_issues": [<list of max 5 specific critical issues>],
    "warnings": [<list of max 5 non-critical warnings>],
    "primary_keyword": "<string>",
    "primary_keyword_density": "<string>",
    "secondary_keywords": [<list of 7 keywords>],
    "long_tail_keywords": [<list of 5 long-tail phrases>],
    "suggested_title": {
        "text": "<string: max 60 chars>",
        "reasoning": "<string: why this title is better>"
    },
    "suggested_meta_description": {
        "text": "<string: max 160 chars>",
        "reasoning": "<string: why this description is better>"
    },
    "suggested_h1": {
        "text": "<string: recommended H1>",
        "reasoning": "<string: why this H1 is better>"
    },
    "suggested_h2s": [
        {"text": "<string>", "reasoning": "<string>"},
        {"text": "<string>", "reasoning": "<string>"}
    ],
    "improved_intro": {
        "text": "<string: rewritten paragraph>",
        "reasoning": "<string: why this is better>"
    },
    "improved_conclusion": {
        "text": "<string: suggested conclusion>",
        "reasoning": "<string: why this is better>"
    },
    "content_gaps": [<list of missing subtopics>],
    "internal_link_suggestions": [<list of 3 pages/topics to link to>],
    "word_count_analysis": {
        "current_estimate": <integer>,
        "recommended_minimum": <integer>,
        "verdict": "<string: 'Too short', 'Good', or 'Comprehensive'>"
    },
    "readability": {
        "level": "<string>",
        "suggestions": [<list of improvement tips>]
    },
    "eeat_analysis": {
        "expertise_signals": "<string>",
        "trust_signals": "<string>",
        "improvement_tips": [<list of suggestions>]
    },
    "action_items": [<list of top 5 prioritized actions>]
}
"""


# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-color: #00CC96;
    }
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    /* Status boxes */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e1e !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOGIN SYSTEM
# ==============================================================================

# Initialize session state for login
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Get PIN from secrets (must be set in .streamlit/secrets.toml or Streamlit Cloud)
# Format in secrets.toml: ACCESS_PIN = "your-pin-here"
ACCESS_PIN = st.secrets.get("ACCESS_PIN", None)

# Login screen
if not st.session_state.authenticated:
    st.title("🔐 SEO Evaluator - Login")
    st.markdown("Please enter the access code to continue.")
    
    # Check if PIN is configured
    if not ACCESS_PIN:
        st.error("❌ ACCESS_PIN not configured in secrets. Please add it to your secrets.toml")
        st.info("""
        **To fix (Streamlit Cloud):**
        1. Go to App Settings → Secrets
        2. Add: `ACCESS_PIN = "your-4-digit-pin"`
        
        **To fix (Local):**
        1. Create `.streamlit/secrets.toml`
        2. Add: `ACCESS_PIN = "your-4-digit-pin"`
        """)
        st.stop()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pin_input = st.text_input(
            "Enter 4-digit PIN",
            type="password",
            max_chars=4,
            placeholder="****"
        )
        
        if st.button("🔓 Login", use_container_width=True):
            if pin_input == ACCESS_PIN:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect PIN. Please try again.")
        
        st.caption("Contact the administrator if you don't have the access code.")
    
    st.stop()  # Stop execution until authenticated


# ==============================================================================
# SIDEBAR CONFIGURATION (Only shown after login)
# ==============================================================================

with st.sidebar:
    st.title("⚙️ AI Configuration")
    
    # Model Selection - Exact models as requested by user
    model_name = st.selectbox(
        "Select Gemini Model",
        [
            "gemini-3-flash-preview",  # Gemini 3 Flash (Preview)
            "gemini-3-pro-preview",    # Gemini 3 Pro (Preview) - CORRECTED
            "gemini-2.5-pro",          # Gemini 2.5 Pro - CORRECTED
        ],
        index=0,  # Default to gemini-3-flash-preview
        help="Choose the AI model for analysis"
    )
    
    # Model Guide
    with st.expander("📖 Which model should I use?"):
        st.markdown("""
        **🚀 gemini-3-flash-preview** (Default)
        - Latest Gemini 3 Flash model
        - Fast & cost-efficient
        - Great for most SEO audits
        
        **💎 gemini-3-pro**
        - Gemini 3 Pro - most powerful
        - Best for complex analysis
        - Use for critical pages
        
        **🔷 gemini-2.5-pro**
        - Stable Pro model
        - Excellent quality
        - Reliable fallback
        """)
    
    st.divider()




    
    # API Key Configuration
    st.subheader("🔑 API Key")
    
    use_custom_api = st.checkbox(
        "Use my own API key",
        value=False,
        help="Check this to use your personal Google AI API key instead of the default."
    )
    
    if use_custom_api:
        custom_api_key = st.text_input(
            "Enter your Google AI API Key",
            type="password",
            placeholder="AIza...",
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )
        
        if custom_api_key and len(custom_api_key) > 10:
            api_key = custom_api_key
            st.success("✅ Using your custom API key")
        else:
            st.warning("⚠️ Please enter a valid API key")
            api_key = None
    else:
        # Use default API key from secrets
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            if api_key and len(api_key) > 10:
                st.success("✅ Using default API key")
            else:
                st.error("❌ Default API key is invalid")
                api_key = None
        else:
            st.warning("⚠️ No default API key configured. Please use your own key.")
            api_key = None
    
    st.divider()
    
    st.info("""
    **How to use:**
    1. Enter a valid URL (http/https).
    2. Wait for crawling & research.
    3. Review the SEO audit & copy improvements.
    
    **Note:** This tool uses live SERP data from DuckDuckGo for keyword research.
    """)
    
    st.divider()
    
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Check if we have a valid API key before proceeding
if not api_key:
    st.error("❌ No valid API key available. Please configure an API key in the sidebar.")
    st.stop()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def validate_url(url: str) -> tuple[bool, str]:
    """
    Validates that the URL has a proper http/https scheme.
    Returns (is_valid, error_message).
    """
    if not url:
        return False, "URL cannot be empty."
    
    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "Invalid URL format - missing domain."
        return True, ""
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"


@st.cache_data(ttl=3600, show_spinner=False)
def crawl_url(url: str) -> tuple[dict | None, str | None]:
    """
    Crawls the URL and extracts content using requests + BeautifulSoup.
    This is more reliable than WebBaseLoader for various sites.
    
    Returns: (data_dict, error_string)
    - data_dict contains: page_content, title, meta_description, h1
    - error_string is None on success, contains error message on failure
    """
    # NOTE: Removed 'br' (Brotli) from Accept-Encoding to avoid binary data issues
    # Only use gzip/deflate which requests handles automatically
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',  # NO brotli - causes binary issues
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        
        # Force encoding detection if not set
        if response.encoding is None or response.encoding == 'ISO-8859-1':
            # Try to detect from content-type header or use apparent_encoding
            response.encoding = response.apparent_encoding or 'utf-8'
        
        # Check for common error codes
        if response.status_code == 403:
            return None, "Access forbidden (403). The website is blocking automated requests."
        elif response.status_code == 404:
            return None, "Page not found (404). Please check the URL."
        elif response.status_code >= 400:
            return None, f"HTTP Error {response.status_code}. Unable to access the page."
        
        # Get text content with proper encoding
        html_content = response.text
        
        # Check if we got binary garbage
        if '\x00' in html_content[:1000] or not any(c.isalpha() for c in html_content[:500]):
            return None, "Received binary or corrupted data. The website may be blocking scrapers."
        
        # Parse HTML with html.parser (most reliable)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title BEFORE removing elements
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No title found"
        
        # Extract meta description BEFORE removing elements
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc_tag:
            meta_desc_tag = soup.find('meta', attrs={'name': 'Description'})
        if not meta_desc_tag:
            meta_desc_tag = soup.find('meta', attrs={'property': 'og:description'})
        meta_description = meta_desc_tag.get('content', '') if meta_desc_tag else "No meta description found"
        
        # Extract H1 BEFORE removing elements
        h1_tag = soup.find('h1')
        h1 = h1_tag.get_text(strip=True) if h1_tag else "No H1 found"
        
        # Now remove non-content elements for text extraction
        for element in soup(['script', 'style', 'noscript', 'iframe', 'svg']):
            element.decompose()
        
        # Extract main text content - try multiple strategies
        page_content = ""
        
        # Strategy 1: Look for main content containers
        content_selectors = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', {'id': 'content'}),
            soup.find('div', {'id': 'main-content'}),
            soup.find('div', {'class': 'content'}),
            soup.find('div', {'class': 'main-content'}),
            soup.find('div', {'role': 'main'}),
        ]
        
        for container in content_selectors:
            if container:
                page_content = container.get_text(separator=' ', strip=True)
                if len(page_content) > 200:
                    break
        
        # Strategy 2: If no main content found, use body
        if len(page_content) < 200:
            body = soup.find('body')
            if body:
                # Remove nav/footer/header from body copy
                body_copy = BeautifulSoup(str(body), 'html.parser')
                for elem in body_copy(['nav', 'footer', 'header', 'aside', 'form']):
                    elem.decompose()
                page_content = body_copy.get_text(separator=' ', strip=True)
        
        # Strategy 3: Fallback to all text
        if len(page_content) < 100:
            page_content = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        page_content = re.sub(r'\s+', ' ', page_content).strip()
        
        # Check for meaningful content
        if len(page_content) < 50:
            return None, "Page content is too short. The page may be mostly JavaScript-rendered."
        
        # Verify we have actual text content (not just symbols/numbers)
        alpha_count = sum(1 for c in page_content[:500] if c.isalpha())
        if alpha_count < 50:
            return None, "Page content appears to be non-text or corrupted."
        
        return {
            'page_content': page_content,
            'title': title,
            'meta_description': meta_description,
            'h1': h1,
            'url': url
        }, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. The website took too long to respond."
    except requests.exceptions.SSLError:
        return None, "SSL certificate error. The website's security certificate is invalid."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Unable to reach the website."
    except Exception as e:
        return None, f"Unexpected error during crawl: {str(e)}"


@st.cache_data(ttl=3600, show_spinner=False)
def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """
    Performs a DuckDuckGo search using multiple parsing strategies.
    Falls back gracefully if scraping fails.
    
    NOTE: This produces SERP-derived keyword ideas, not authoritative keyword volume data.
    """
    results = []
    
    try:
        # Use DuckDuckGo HTML search with proper headers
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://duckduckgo.com/',
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Strategy 1: Try result__body class
            result_elements = soup.find_all('div', class_='result__body')[:max_results]
            
            # Strategy 2: Try result class if strategy 1 fails
            if not result_elements:
                result_elements = soup.find_all('div', class_='result')[:max_results]
            
            # Strategy 3: Try links-of-type pattern
            if not result_elements:
                result_elements = soup.find_all('div', class_='links_main')[:max_results]
            
            for elem in result_elements:
                # Try multiple title selectors
                title_elem = (
                    elem.find('a', class_='result__a') or 
                    elem.find('a', class_='result-link') or
                    elem.find('h2') or
                    elem.find('a')
                )
                
                # Try multiple snippet selectors  
                snippet_elem = (
                    elem.find('a', class_='result__snippet') or
                    elem.find('div', class_='result__snippet') or
                    elem.find('span', class_='result__snippet') or
                    elem.find('p')
                )
                
                if title_elem:
                    title_text = title_elem.get_text(strip=True)
                    if title_text and len(title_text) > 3:
                        results.append({
                            'title': title_text,
                            'link': title_elem.get('href', ''),
                            'snippet': snippet_elem.get_text(strip=True)[:200] if snippet_elem else ''
                        })
        
    except Exception as e:
        # Log but don't fail
        pass
    
    return results


def get_research_keywords(llm, page_content_snippet: str) -> tuple[str, str, list[dict]]:
    """
    Uses Gemini to identify the topic and generate keyword suggestions.
    Also attempts DuckDuckGo search for real-time SERP signals.
    
    If DuckDuckGo fails, Gemini will generate keywords based on the content alone.
    
    Returns: (topic_string, research_text, source_links)
    """
    # Step 1: Use Gemini to identify the main topic (3-6 words)
    topic_prompt = PromptTemplate.from_template(
        "Extract the main topic of this text in 3-6 words maximum. Only output the topic, nothing else.\n\nText: {text}"
    )
    topic_chain = topic_prompt | llm | StrOutputParser()
    
    try:
        topic = topic_chain.invoke({"text": page_content_snippet[:2000]}).strip()
        topic = topic.strip('"\'').strip()
    except Exception as e:
        topic = "general topic"
    
    # Step 2: Try DuckDuckGo searches
    search_queries = [
        f"{topic} keywords",
        f"{topic} best practices 2026",
        f"{topic} tips"
    ]
    
    all_results = []
    research_parts = []
    source_links = []
    
    for query in search_queries:
        results = search_duckduckgo(query, max_results=3)
        
        if results:
            research_parts.append(f"**Query:** {query}")
            for r in results:
                research_parts.append(f"- {r['title']}: {r['snippet'][:100]}...")
                source_links.append({'title': r['title'], 'link': r['link']})
            research_parts.append("")
            all_results.extend(results)
    
    # Step 3: If DuckDuckGo failed, use Gemini to generate keyword ideas
    if not all_results:
        keyword_prompt = PromptTemplate.from_template(
            """Based on this content about "{topic}", suggest 10 relevant SEO keywords.
            Format: One keyword per line, no numbering, no explanations.
            
            Content snippet: {content}"""
        )
        keyword_chain = keyword_prompt | llm | StrOutputParser()
        
        try:
            gemini_keywords = keyword_chain.invoke({
                "topic": topic,
                "content": page_content_snippet[:1500]
            })
            
            research_parts.append("**AI-Generated Keywords** (DuckDuckGo unavailable):")
            for kw in gemini_keywords.strip().split('\n')[:10]:
                kw = kw.strip().strip('-').strip('•').strip()
                if kw:
                    research_parts.append(f"- {kw}")
                    source_links.append({'title': kw, 'link': ''})
            research_parts.append("")
            research_parts.append("*Note: Keywords generated by AI based on page content.*")
        except:
            research_parts.append("**Note:** Could not fetch live keyword data.")
    
    research_text = "\n".join(research_parts) if research_parts else "No research data available."
    
    return topic, research_text, source_links



def analyze_content(llm, page_data: dict, research_data: str) -> str:
    """
    Main analysis function. Sends page content + research to Gemini for SEO analysis.
    
    Token usage guardrail: Content is truncated to avoid excessive token usage.
    """
    
    prompt = PromptTemplate.from_template(
        """
{system_prompt}

--- INPUT DATA ---

PAGE URL: {url}
PAGE TITLE: {title}
PAGE H1: {h1}
META DESCRIPTION: {meta_description}

LIVE KEYWORD RESEARCH FROM SERP:
{research}

PAGE CONTENT (First 15000 characters):
{content}

--- END INPUT ---

Analyze this page for SEO and output ONLY valid JSON matching the schema above.
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Truncate content to manage token usage (Gemini has large context but we want speed)
    truncated_content = page_data['page_content'][:15000]
    
    return chain.invoke({
        "system_prompt": GEMINI_SYSTEM_PROMPT,
        "url": page_data.get('url', 'N/A'),
        "title": page_data.get('title', 'N/A'),
        "h1": page_data.get('h1', 'N/A'),
        "meta_description": page_data.get('meta_description', 'N/A'),
        "research": research_data,
        "content": truncated_content
    })


def parse_gemini_response(response_text: str) -> dict:
    """
    Safely parses the Gemini JSON response.
    Advanced regex to handle markdown, preambles, and mixed content from Pro models.
    """
    try:
        # 1. Clean up known markdown wrapper patterns
        text = response_text.strip()
        
        # 2. Use regex to find the largest outer JSON object
        # Pattern looks for { ... } across multiple lines
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        
        # Fallback simplistic regex if recursive regex fails or isn't supported
        if not json_match:
             json_match = re.search(r'\{[\s\S]*\}', text)
             
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # If no JSON structure found, raise error
            raise ValueError("No JSON object found in response")
            
    except (json.JSONDecodeError, ValueError) as e:
        # If parsing fails, try to "fix" common issues
        try:
            # Sometimes models return single quotes instead of double
            fixed_text = text.replace("'", '"')
            json_match = re.search(r'\{[\s\S]*\}', fixed_text)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        raise e


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

st.title("📊 Advanced SEO Content Evaluator")
st.markdown("Enter a URL to audit its SEO performance and get AI-driven improvements based on live search data.")

# URL Input
url_input = st.text_input(
    "Enter URL to Analyze",
    placeholder="https://example.com/your-page",
    help="Enter the full URL including http:// or https://"
)

# Main analysis flow
if url_input:
    # Validate URL
    is_valid, error_msg = validate_url(url_input)
    
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    # Initialize Gemini LLM
    try:
        chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,  # Uses custom or default API key from sidebar
            temperature=0.3,  # Low temperature for consistent, factual output
            max_output_tokens=4096
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini model: {str(e)}")
        st.info("Please verify your API key is valid and the model name is correct.")
        st.stop()
    
    # =========================================================================
    # PHASE 1: CRAWL
    # =========================================================================
    with st.status("🔍 Phase 1: Crawling Website...", expanded=True) as status:
        page_data, crawl_error = crawl_url(url_input)
        
        if crawl_error:
            status.update(label="❌ Crawl Failed!", state="error", expanded=True)
            st.error(f"Error: {crawl_error}")
            st.stop()
        
        # Show extracted data summary
        st.write(f"**Title:** {page_data['title'][:80]}...")
        st.write(f"**H1:** {page_data['h1'][:80]}...")
        st.write(f"**Content Length:** {len(page_data['page_content']):,} characters")
        
        status.update(label="✅ Website Crawled Successfully!", state="complete", expanded=False)
    
    # =========================================================================
    # PHASE 2: RESEARCH
    # =========================================================================
    with st.status("🌐 Phase 2: Live Keyword Research...", expanded=True) as status:
        topic, research_text, source_links = get_research_keywords(chat_model, page_data['page_content'])
        
        st.write(f"**Identified Topic:** {topic}")
        
        with st.expander("View Raw Search Data"):
            st.text(research_text)
        
        status.update(label="✅ Research Completed!", state="complete", expanded=False)
    
    # =========================================================================
    # PHASE 3: GEMINI ANALYSIS
    # =========================================================================
    with st.spinner("🤖 Phase 3: Gemini is Analyzing SEO & Writing Improvements..."):
        try:
            analysis_response = analyze_content(chat_model, page_data, research_text)
            result = parse_gemini_response(analysis_response)
            
        except json.JSONDecodeError as e:
            st.error("❌ Error parsing AI response. The model returned invalid JSON.")
            with st.expander("Debug: Raw AI Output"):
                st.code(analysis_response, language="text")
            st.info("Try clicking 'Analyze' again, or try a different Gemini model.")
            st.stop()
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.stop()
    
    # =========================================================================
    # PHASE 4: RESULTS DISPLAY (Enhanced for Reasoning)
    # =========================================================================
    st.divider()
    st.subheader("📈 Results & Recommendations")
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = result.get('seo_score', 0)
        st.metric("SEO Score", f"{score}/100")
        st.progress(min(score / 100, 1.0))
        
    with col2:
        st.write("**Page Intent**")
        st.info(result.get('page_intent', 'General'))
        
    with col3:
        st.write("**Primary Keyword**")
        st.caption(result.get('primary_keyword', 'N/A'))
        
    with col4:
        word_count = result.get('word_count_analysis', {})
        st.write("**Word Count**")
        st.caption(f"{word_count.get('current_estimate', 'N/A')} words ({word_count.get('verdict', 'N/A')})")
    
    # Score Breakdown
    score_breakdown = result.get('score_breakdown', {})
    if score_breakdown:
        st.subheader("📊 Score Breakdown")
        sb_col1, sb_col2, sb_col3, sb_col4 = st.columns(4)
        with sb_col1:
            st.metric("Keywords", f"{score_breakdown.get('keyword_optimization', 0)}/25")
        with sb_col2:
            st.metric("Content", f"{score_breakdown.get('content_quality', 0)}/25")
        with sb_col3:
            st.metric("Technical", f"{score_breakdown.get('technical_seo', 0)}/25")
        with sb_col4:
            st.metric("UX", f"{score_breakdown.get('user_experience', 0)}/25")
    
    # Critical Issues & Warnings
    issues_col1, issues_col2 = st.columns(2)
    
    with issues_col1:
        st.subheader("🚨 Critical Issues")
        critical_issues = result.get("critical_issues", [])
        if critical_issues:
            for issue in critical_issues:
                st.markdown(f"- 🔴 {issue}")
        else:
            st.success("✅ No critical issues!")
            
    with issues_col2:
        st.subheader("⚠️ Warnings")
        warnings = result.get("warnings", [])
        if warnings:
            for warning in warnings:
                st.markdown(f"- 🟠 {warning}")
        else:
            st.success("✅ No warnings!")
            
    # Priority Action Items
    st.subheader("🎯 Priority Action Items")
    action_items = result.get("action_items", [])
    if action_items:
        for i, action in enumerate(action_items, 1):
            st.markdown(f"**{i}.** {action}")
    else:
        st.write("No action items generated.")

    st.divider()
    
    st.header("✨ Targeted Improvements")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏷️ Meta Tags", 
        "📝 Content", 
        "🔑 Keywords",
        "🏆 EEAT & Trust", 
        "📖 Readability", 
        "🔗 Sources"
    ])
    
    with tab1:
        st.subheader("Title & Meta Description")
        
        # Display with Reasoning
        s_title = result.get('suggested_title', {})
        st.write("**Suggested Title:**")
        if isinstance(s_title, dict):
            st.code(s_title.get('text', 'N/A'), language='html')
            if s_title.get('reasoning'):
                st.caption(f"💡 *Why: {s_title.get('reasoning')}*")
        else:
            st.code(s_title, language='html')
            
        s_meta = result.get('suggested_meta_description', {})
        st.write("**Suggested Meta Description:**")
        if isinstance(s_meta, dict):
            st.code(s_meta.get('text', 'N/A'), language='html')
            if s_meta.get('reasoning'):
                st.caption(f"💡 *Why: {s_meta.get('reasoning')}*")
        else:
            st.code(s_meta, language='html')
            
        s_h1 = result.get('suggested_h1', {})
        st.write("**Suggested H1:**")
        if isinstance(s_h1, dict):
            st.code(s_h1.get('text', 'N/A'), language='html')
            if s_h1.get('reasoning'):
                st.caption(f"💡 *Why: {s_h1.get('reasoning')}*")
        else:
            st.code(s_h1, language='html')
            
        st.subheader("Suggested H2s")
        for h2 in result.get('suggested_h2s', []):
            if isinstance(h2, dict):
                st.markdown(f"- **{h2.get('text')}**")
                st.caption(f"  └─ *{h2.get('reasoning')}*")
            else:
                st.markdown(f"- {h2}")

    with tab2:
        st.subheader("Content Optimization")
        
        s_intro = result.get('improved_intro', {})
        with st.expander("📄 Rewritten Introduction (Optimized)", expanded=True):
            if isinstance(s_intro, dict):
                st.markdown(s_intro.get('text', 'N/A'))
                if s_intro.get('reasoning'):
                    st.caption(f"💡 *Why: {s_intro.get('reasoning')}*")
            else:
                st.markdown(s_intro)
                
        s_conc = result.get('improved_conclusion', {})
        with st.expander("🔚 Suggested Conclusion"):
            if isinstance(s_conc, dict):
                st.markdown(s_conc.get('text', 'N/A'))
                if s_conc.get('reasoning'):
                    st.caption(f"💡 *Why: {s_conc.get('reasoning')}*")
            else:
                st.markdown(s_conc)
                
        st.subheader("Content Gaps")
        for gap in result.get('content_gaps', []):
            st.markdown(f"- 📌 {gap}")
            
        st.subheader("Internal Linking Opportunities")
        for link in result.get('internal_link_suggestions', []):
            st.markdown(f"- 🔗 {link}")

    with tab3:
        st.subheader("Keyword Strategy")
        st.table({
            "Metric": ["Primary Keyword", "Density"],
            "Value": [result.get('primary_keyword'), result.get('primary_keyword_density')]
        })
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.write("**Secondary Keywords**")
            for kw in result.get('secondary_keywords', []):
                st.markdown(f"- {kw}")
        with col_k2:
            st.write("**Long-Tail Opportunities**")
            for kw in result.get('long_tail_keywords', []):
                st.markdown(f"- {kw}")

    
    with tab4:
        eeat = result.get('eeat_analysis', {})
        if eeat:
            st.markdown("### Expertise Signals")
            st.write(eeat.get('expertise_signals', 'N/A'))
            
            st.markdown("### Trust Signals")
            st.write(eeat.get('trust_signals', 'N/A'))
            
            st.markdown("### EEAT Improvement Tips")
            for tip in eeat.get('improvement_tips', []):
                st.markdown(f"- 💡 {tip}")
        else:
            st.write("EEAT analysis not available.")
    
    with tab5:
        readability = result.get('readability', {})
        if readability:
            st.markdown(f"### Readability Level: **{readability.get('level', 'N/A')}**")
            
            st.markdown("### Improvement Suggestions")
            for suggestion in readability.get('suggestions', []):
                st.markdown(f"- ✏️ {suggestion}")
        
        word_count = result.get('word_count_analysis', {})
        if word_count:
            st.markdown("### Word Count Analysis")
            st.write(f"**Current:** ~{word_count.get('current_estimate', 'N/A')} words")
            st.write(f"**Recommended Minimum:** {word_count.get('recommended_minimum', 'N/A')} words")
            st.write(f"**Verdict:** {word_count.get('verdict', 'N/A')}")
    
    with tab6:
        st.markdown("### Research Source Links")
        st.caption("These links were used to derive keyword ideas:")
        if source_links:
            for link in source_links[:10]:
                if link.get('link'):
                    st.markdown(f"- [{link['title'][:50]}...]({link['link']})")
        else:
            st.write("No source links available.")
    
    # PageSpeed Reminder
    st.divider()
    st.info("📱 **Reminder:** Test your page's mobile speed at [PageSpeed Insights](https://pagespeed.web.dev/)")
    
    # Debug Section
    with st.expander("🔧 Debug: Raw JSON Response"):
        st.json(result)

