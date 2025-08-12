# 🚀 PRD Creator - Enterprise AI Product Documentation Generator

Enterprise-grade product requirements document generator powered by 10 specialized AI agents. Transform your product idea into comprehensive professional documentation.

## ✨ Features

### 🤖 10 Expert AI Agents
- **🎨 Brand Strategist** - Comprehensive brand identity & design systems
- **🏗️ Principal Architect** - Enterprise technical architecture & AI integration
- **📋 Senior PM** - MoSCoW prioritized features with scope boundaries
- **📑 VP Product** - Risk analysis & competitive intelligence  
- **📅 Program Manager** - Timeline & dependency mapping
- **💼 Business Analyst** - Market analysis & ROI projections
- **🎨 UX/UI Designer** - Wireframes, user flows & component library
- **🧪 QA Test Architect** - Comprehensive test planning & automation
- **🗄️ Data Architect** - Database schemas & API contracts
- **🚀 DevOps Engineer** - CI/CD pipelines & infrastructure automation
- **🗂️ Project Manager** - IDE-compatible Kanban tasks

### 🔒 Security and Performance
- ✅ API key validation
- ✅ Input validation and sanitization
- ✅ Rate limiting protection
- ✅ HTTP timeout settings
- ✅ Comprehensive error handling
- ✅ Progress tracking

### 📤 Output Formats
- **Comprehensive Markdown** - Complete product documentation
- **10 Specialized JSON Files** - Structured data for each domain
  - `branding.json` - Brand identity & design systems
  - `technical.json` - Architecture & AI integration specs  
  - `features.json` - MoSCoW prioritized features + P3 scope
  - `prd.json` - Product requirements & competitive analysis
  - `timeline.json` - Project phases & dependency mapping
  - `business_case.json` - Market analysis & ROI projections
  - `uiux_design.json` - Wireframes & component library
  - `test_plan.json` - Comprehensive testing strategies
  - `data_architecture.json` - Database schemas & API contracts
  - `devops_pipeline.json` - CI/CD & infrastructure automation
- **IDE Tasks** - Development-ready Kanban board

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/palamut62/prd-creator-ai.git
cd prd-creator-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
```bash
cp .env.example .env
```

Edit the `.env` file and add your API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Run the Application
```bash
streamlit run main.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | - | OpenRouter API key (required) |
| `MODEL_NAME` | `openai/gpt-5` | AI model to use |
| `MAX_REQUESTS_PER_WINDOW` | `5` | Rate limit: maximum request count |
| `RATE_LIMIT_WINDOW_SECONDS` | `300` | Rate limit: time window (seconds) |
| `DEFAULT_TIMEOUT_SECONDS` | `60` | API request timeout duration |
| `OUTPUT_DIR` | `outputs` | Directory to save output files |

### Supported Models
**🆓 Free Models:**
- `openai/gpt-oss-20b:free` - GPT-OSS 20B
- `z-ai/glm-4.5-air:free` - GLM 4.5 Air  
- `qwen/qwen3-coder:free` - Qwen3 Coder

**⚡ Performance Models:**
- `openai/gpt-4o` - GPT-4o ($2.50/1M)
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet ($3/1M)
- `google/gemini-2.5-pro` - Gemini 2.5 Pro ($3.50/1M)

**🚀 Premium Models:**
- `openai/gpt-5` - GPT-5 ($25/1M)
- `anthropic/claude-3-opus` - Claude 3 Opus ($15/1M)

## 📋 Usage

1. **Enter Product Idea**: Describe your product idea in detail on the main page
2. **Generate Documents**: Click the "Generate Documents and IDE Tasks" button
3. **Download Results**: Download the generated documents or save them as files

### Tips
- The more detailed you are, the better documents will be generated
- Minimum 10 characters, maximum 5000 character limit
- Rate limiting allows maximum 5 requests per 5 minutes

## 🏗️ Project Structure

```
PRD-Creator-AI/
├── main.py              # Main Streamlit application (2300+ lines)
├── config.py            # Configuration settings & model definitions
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template  
├── README.md           # Project documentation
└── outputs/            # Generated documents (auto-created)
    ├── product_docs_[timestamp].md     # Complete documentation
    ├── branding_[timestamp].json       # Brand identity specs
    ├── technical_[timestamp].json      # Architecture & AI specs
    ├── features_[timestamp].json       # Features & scope boundaries
    ├── prd_[timestamp].json           # Product requirements
    ├── timeline_[timestamp].json       # Project timeline
    ├── business_case_[timestamp].json  # Market & ROI analysis
    ├── uiux_design_[timestamp].json   # Wireframes & components
    ├── test_plan_[timestamp].json     # Testing strategies
    ├── data_architecture_[timestamp].json # DB & API specs
    ├── devops_pipeline_[timestamp].json   # CI/CD automation
    └── dev_tasks_[timestamp].md       # IDE Kanban tasks
```

## 🔍 API Testing

You can test the API connection in the application:
1. Click the "Test API Connection" button from the sidebar
2. Check the connection status

## ⚠️ Troubleshooting

### Common Errors

**API Key Error**
- Ensure the `OPENROUTER_API_KEY` value in the `.env` file is correct
- Check that the API key is in valid format

**Rate Limit Error**  
- Wait 5 minutes and try again
- Check your request count (shown in the sidebar)

**Timeout Error**
- Check your internet connection
- Try a shorter product idea description
- Increase the `DEFAULT_TIMEOUT_SECONDS` value

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues:
1. First check this README
2. Search in GitHub Issues
3. Create a new issue