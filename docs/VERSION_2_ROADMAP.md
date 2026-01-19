# SEO Agent - Version 2.0 Roadmap

> **Target Stack**: Next.js + FastAPI  
> **Status**: Planned  
> **Last Updated**: January 2026

---

## Architecture

```
┌─────────────────────────────────────────┐
│        Streamlit Cloud (Free)           │
│  ┌─────────────────────────────────┐    │
│  │        Python + Streamlit       │    │
│  │        Enhanced UI/CSS          │    │
│  │        Gemini + OpenAI APIs     │    │
│  │        DuckDuckGo Search        │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Keeping existing Streamlit stack with UI enhancements.**

---

## New Features

### Multi-Provider AI Support

| Provider | Available Models |
|----------|------------------|
| Google Gemini | gemini-3-flash-preview, gemini-3-pro-preview, gemini-2.5-pro |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o1-mini |

- User provides own API key
- Model selection dropdown

### API Cost Tracking

- Token usage per request (input + output)
- Estimated cost calculation
- Cumulative usage display in sidebar

### Enhanced Chatbot

| Feature | Description |
|---------|-------------|
| Floating Button | "💬" fixed to bottom-right |
| Modal Chat | Opens inline without leaving page |
| Full-Screen | Dedicated page for immersive chat |
| Web Search | DuckDuckGo search during conversation |
| Context | Reads SEO analysis results |
| Quick Actions | Pre-built prompts |

### Premium UI

- Modern, minimal, premium aesthetic
- Card-based layout with soft shadows
- GSAP animations (stagger, scroll reveal)
- Radial SEO score gauge with animation
- Dark/Light theme toggle
- Responsive (desktop/tablet/mobile)

---

## Hosting

- **Current**: Streamlit Cloud (free)
- **No changes planned** - keeping existing deployment

---

## Code Reuse

| Component | Changes |
|-----------|---------|
| Gemini API calls | None |
| DuckDuckGo search | None |
| JSON parsing | None |
| System prompts | None |
| Streamlit UI | **Replaced** with Next.js |
| Session state | **Replaced** with API endpoints |

**~90% of backend logic stays identical**

---

## UI/UX Improvements

| Current (Streamlit) | New (Next.js) |
|---------------------|---------------|
| Limited layout control | Full CSS Grid/Flexbox |
| No animations | GSAP + ScrollTrigger |
| Basic components | shadcn/ui + custom |
| Hacky floating chat | Native modal |
| Basic theme toggle | Full theme system |

---

## Development Phases

1. **Phase 1**: Convert Python → FastAPI backend
2. **Phase 2**: Build Next.js frontend shell
3. **Phase 3**: Implement Evaluator UI
4. **Phase 4**: Implement Chat UI
5. **Phase 5**: Add OpenAI support
6. **Phase 6**: Add cost tracking
7. **Phase 7**: Deploy to Vercel + Render
