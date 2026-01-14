import streamlit as st
import time
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(
    page_title="Advanced SEO Evaluator AI",
    page_icon="🚀",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00CC96;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
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
</style>
""", unsafe_allow_html=True)

# --- Sidebar & Configuration ---
with st.sidebar:
    st.title("⚙️ AI Configuration")
    
    # Model Selection
    model_name = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"],
        index=0,
        help="Gemini 2.5 Flash is the current standard. Version 3 models are in preview."
    )
    
    st.info("""
    **How to use:**
    1. Enter a valid URL.
    2. Wait for the AI to crawl & research.
    3. Review the Audit & Improvements.
    
    **Note:** This tool uses real-time SERP data via DuckDuckGo.
    """)
    
    # API Key check
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("API Key loaded successfully")
    else:
        st.error("Missing GOOGLE_API_KEY in secrets. Please verify setup.")
        st.stop()

# --- Functions ---

@st.cache_data(ttl=3600, show_spinner=False)
def crawl_url(url):
    """Crawls the URL and extracts content."""
    try:
        loader = WebBaseLoader(
            url,
            header_template={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        docs = loader.load()
        if not docs:
            return None, "No content found on the page."
            
        doc = docs[0]
        # Basic extraction improvements could happen here if needed using BS4 directly
        # but WebBaseLoader is usually sufficient for text.
        return doc, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def get_research_keywords(_llm, page_content_snippet):
    """Uses LLM to topicize + DuckDuckGo to find keywords."""
    
    # 1. Identify Topic
    topic_prompt = PromptTemplate.from_template(
        "Extract the main topic of this text in 2-4 keywords maximum. Text: {text}"
    )
    topic_chain = topic_prompt | _llm | StrOutputParser()
    try:
        # Use first 2000 chars for topic extraction to save speed/tokens
        topic = topic_chain.invoke({"text": page_content_snippet[:2000]}).strip()
    except Exception as e:
        topic = "general topic"

    # 2. Perform Searches
    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="y", max_results=5)
    search = DuckDuckGoSearchRun(api_wrapper=wrapper)
    
    # Run searches (could be parallelized but sequential is safer for rate limits)
    search_queries = [
        f"primary keyword for {topic}",
        f"{topic} best practices SEO",
        f"people also ask about {topic}"
    ]
    
    results = []
    for q in search_queries:
        try:
            res = search.run(q)
            results.append(f"**Query:** {q}\n**Result:** {res}")
        except Exception:
            results.append(f"**Query:** {q}\n**Result:** Failed to fetch")
            
    return topic, "\n\n".join(results)

def analyze_content(_llm, content, metadata, research_data):
    """Main analysis chain."""
    
    system_instruction = """
    You are an elite SEO specialist using modern ranking factors (EEAT, Helpfulness). 
    You will receive page content and live keyword ideas from search results.
    
    Your task:
    1. Identify the best Primary Keyword based on the research.
    2. Analyze the current content against this keyword (Usage in Title, H1, First 100 words, Density).
    3. Generate specific improvements.
    
    OUTPUT SCHEMA (JSON ONLY):
    {
        "seo_score": <integer 0-100>,
        "critical_issues": [<list of strings, max 3 critical specific issues>],
        "primary_keyword": "<string>",
        "secondary_keywords": [<list of 5 strings>],
        "suggested_title": "<string, max 60 chars, include keyword>",
        "suggested_meta_description": "<string, max 160 chars, compelling call to action>",
        "suggested_h1": "<string>",
        "improved_intro": "<string, rewrite first paragraph to naturally include primary keyword>",
        "content_gaps": [<list of missing subtopics based on research>]
    }
    """
    
    prompt = PromptTemplate.from_template(
        """
        {system_instruction}
        
        --- INPUT DATA ---
        
        PAGE TITLE: {title}
        PAGE METADATA: {meta}
        
        RESEARCH / SERP INSIGHTS:
        {research}
        
        PAGE CONTENT (Truncated):
        {content}
        
        --- END INPUT ---
        
        Provide the JSON analysis now.
        """
    )
    
    chain = prompt | _llm | StrOutputParser()
    
    # Truncate content to avoid exceeding context window if massive, though 2.0 has huge window
    # Safe limit for reasonable processing time
    truncated_content = content[:30000] 
    
    return chain.invoke({
        "system_instruction": system_instruction,
        "title": metadata.get('title', 'N/A'),
        "meta": metadata.get('description', 'N/A'),
        "research": research_data,
        "content": truncated_content
    })

# --- Main App Interface ---

st.title("📊 Advanced SEO Content Evaluator")
st.markdown("Enter a URL to audit its SEO performance and get AI-driven improvements based on clear live search data.")

url_input = st.text_input("Enter URL to Analyze", placeholder="https://example.com/my-page")

if url_input:
    if not url_input.startswith("http"):
        st.error("Please enter a valid URL (starting with http:// or https://)")
    else:
        # Initialize LLM
        try:
            chat_model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0.3
            )
        except Exception as e:
            st.error(f"Failed to initialize model. Check API Key. Error: {e}")
            st.stop()
            
        # --- PHASE 1: CRAWL ---
        with st.status("🔍 Phase 1: Crawling Website...", expanded=True) as status:
            doc, error = crawl_url(url_input)
            
            if error:
                status.update(label="❌ Crawl Failed!", state="error")
                st.error(f"Error: {error}")
                st.stop()
            
            status.update(label="✅ Website Crawled Successfully!", state="complete", expanded=False)
            
        # --- PHASE 2: RESEARCH ---
        with st.status("🌐 Phase 2: Live Keyword Research...", expanded=True) as status:
            topic, research_text = get_research_keywords(chat_model, doc.page_content)
            status.write(f"**Identified Topic:** {topic}")
            with st.expander("View Raw Search Data"):
                st.text(research_text)
            status.update(label="✅ Research Completed!", state="complete", expanded=False)
            
        # --- PHASE 3: ANALYSIS ---
        with st.spinner("🤖 Phase 3: Gemini is Analyzing SEO & Writing Improvements..."):
            try:
                # Prepare metadata (WebBaseLoader sometimes puts title in metadata)
                meta_info = doc.metadata
                analysis_json_str = analyze_content(chat_model, doc.page_content, meta_info, research_text)
                
                # Sanitize code blocks if the model wrapped it
                clean_json = analysis_json_str.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_json)
                
            except json.JSONDecodeError:
                st.error("Error parsing AI response. Please try again.")
                with st.expander("Raw Output"):
                    st.text(analysis_json_str)
                st.stop()
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.stop()

        # --- PHASE 4: REPORT ---
        st.divider()
        st.subheader("Results & Recommendations")
        
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SEO Score", f"{result.get('seo_score', 0)}/100")
            st.progress(result.get('seo_score', 0) / 100)
        with col2:
            st.write("**Primary Keyword**")
            st.info(result.get('primary_keyword', 'N/A'))
        with col3:
            st.write("**Topic**")
            st.caption(topic)

        # Critical Issues
        st.subheader("🚨 Critical Issues")
        if result.get("critical_issues"):
            for issue in result["critical_issues"]:
                st.markdown(f"- 🔴 {issue}")
        else:
            st.success("No critical issues found!")

        # Content Improvements
        st.subheader("✍️ Content Improvements")
        
        tab1, tab2, tab3 = st.tabs(["Meta Data", "Intro Rewrite", "Keywords & Gaps"])
        
        with tab1:
            st.markdown("### Title Tag")
            st.code(result.get('suggested_title'), language='html')
            st.caption(f"Reasoning: Targeting '{result.get('primary_keyword')}' within 60 chars.")
            
            st.markdown("### Meta Description")
            st.code(result.get('suggested_meta_description'), language='html')
            
            if result.get('suggested_h1'):
                st.markdown("### H1 Header")
                st.code(result.get('suggested_h1'), language='html')

        with tab2:
            st.markdown("### Optimized Introduction")
            st.write("Replace your current first paragraph with this version which naturally includes keywords:")
            st.info(result.get('improved_intro'))
            
        with tab3:
            st.markdown("**Secondary Keywords to target:**")
            st.write(", ".join([f"`{k}`" for k in result.get('secondary_keywords', [])]))
            
            st.markdown("**Content Gaps (Add sections on these):**")
            for gap in result.get('content_gaps', []):
                st.markdown(f"- {gap}")

        # Debug Data
        with st.expander("See Raw JSON Response"):
            st.json(result)
