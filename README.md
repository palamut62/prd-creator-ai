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

### 🔒 Güvenlik ve Performans
- ✅ API anahtarı doğrulama
- ✅ Input validation ve sanitization
- ✅ Rate limiting koruması
- ✅ HTTP timeout ayarları
- ✅ Kapsamlı hata yönetimi
- ✅ Progress tracking

### 📤 Çıktı Formatları
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

## 🚀 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone <repo-url>
cd Prd_creator
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarlayın
```bash
cp .env.example .env
```

`.env` dosyasını düzenleyip API anahtarınızı ekleyin:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Uygulamayı Çalıştırın
```bash
streamlit run main.py
```

## 🔧 Konfigürasyon

### Çevre Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `OPENROUTER_API_KEY` | - | OpenRouter API anahtarı (gerekli) |
| `MODEL_NAME` | `openai/gpt-5` | Kullanılacak AI modeli |
| `MAX_REQUESTS_PER_WINDOW` | `5` | Rate limit: maksimum istek sayısı |
| `RATE_LIMIT_WINDOW_SECONDS` | `300` | Rate limit: zaman penceresi (saniye) |
| `DEFAULT_TIMEOUT_SECONDS` | `60` | API isteği timeout süresi |
| `OUTPUT_DIR` | `outputs` | Çıktı dosyalarının kaydedileceği dizin |

### Desteklenen Modeller
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

## 📋 Kullanım

1. **Ürün Fikrini Girin**: Ana sayfada ürün fikrinizi detaylı açıklayın
2. **Dokümanları Oluşturun**: "Doküman Oluştur" butonuna tıklayın
3. **Sonuçları İndirin**: Oluşturulan dokümanları indirin veya dosya olarak kaydedin

### İpuçları
- Ne kadar detaylı açıklarsanız o kadar iyi dokümanlar üretilir
- Minimum 10 karakter, maksimum 5000 karakter sınırı vardır
- Rate limiting nedeniyle 5 dakikada maksimum 5 istek gönderebilirsiniz

## 🏗️ Proje Yapısı

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

## 🔍 API Test Etme

Uygulamada API bağlantısını test edebilirsiniz:
1. Sol panelden "API Bağlantısı Test Et" butonuna tıklayın
2. Bağlantı durumunu kontrol edin

## ⚠️ Sorun Giderme

### Yaygın Hatalar

**API Anahtarı Hatası**
- `.env` dosyasında `OPENROUTER_API_KEY` değerinin doğru olduğundan emin olun
- API anahtarının geçerli formatta olduğunu kontrol edin

**Rate Limit Hatası**  
- 5 dakika bekleyip tekrar deneyin
- İstek sayınızı kontrol edin (sol panelde gösterilir)

**Timeout Hatası**
- İnternet bağlantınızı kontrol edin
- Daha kısa ürün fikri tanımı yapmayı deneyin
- `DEFAULT_TIMEOUT_SECONDS` değerini artırın

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik ekle'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🆘 Destek

Sorunlarınız için:
1. Önce bu README'yi kontrol edin
2. GitHub Issues'da arama yapın
3. Yeni bir issue oluşturun