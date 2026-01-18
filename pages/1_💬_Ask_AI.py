import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(
    page_title="Ask AI - SEO Assistant",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Ask AI About Your Analysis")

# Check if we have analysis data
if 'analysis_result' not in st.session_state or 'page_data' not in st.session_state:
    st.warning("⚠️ No analysis data found. Please go to the **Home** page and run an analysis first.")
    st.info("Once the analysis is complete, you can come back here to ask questions concerning the results.")
    st.switch_page("app.py") # Optional: Redirect button or just let them navigate manually
    st.stop()

# Get data from session state
result = st.session_state['analysis_result']
page_data = st.session_state['page_data']
research_text = st.session_state.get('research_text', "No research data.")
api_key = st.session_state.get('api_key')
model_name = st.session_state.get('model_name', 'gemini-3-flash-preview')

if not api_key:
    st.error("❌ API Key missing. Please login/configure on the Home page.")
    st.stop()

# Initialize Chat Model
try:
    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.4,
        max_output_tokens=2048
    )
except Exception as e:
    st.error(f"❌ Failed to initialize AI: {str(e)}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Add initial system context (invisible to user interface, but priming the model)
    # We'll actually inject this into the context of every prompt or as a SystemMessage if supported/managed manually.
    # For simplicity with the standard chat interface, we'll construct the prompt on the fly.

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your SEO score or improvements..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Construct Context
            context_prompt = f"""
You are an expert SEO data analyst assisting a user with their recent website audit.
Use the following analysis data to answer the user's question.

--- AUDIT CONTEXT ---
URL: {page_data.get('url')}
Title: {page_data.get('title')}
SEO Score: {result.get('seo_score')}/100
Intent Identified: {result.get('page_intent')}
Primary Keyword: {result.get('primary_keyword')}

CRITICAL ISSUES:
{", ".join(result.get('critical_issues', []))}

ACTION ITEMS:
{", ".join(result.get('action_items', []))}

FULL ANALYSIS DATA:
{result}

RESEARCH CONTEXT:
{research_text[:2000]}

PAGE CONTENT SNIPPET:
{page_data.get('page_content', '')[:3000]}
--- END CONTEXT ---

USER QUESTION: {prompt}

Provide a helpful, specific, and actionable answer based on the audit data above. 
If the user asks for a rewrite, use the page content and research to provide a high-quality draft.
Be conversational but professional.
"""
            
            try:
                # We use invoke directly here for simplicity
                ai_response = chat_model.invoke(context_prompt).content
                st.markdown(ai_response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
