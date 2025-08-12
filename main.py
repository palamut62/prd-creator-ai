import streamlit as st
import json
import asyncio
import aiohttp
import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict
import re
from pathlib import Path
from datetime import datetime
import html
import unicodedata
import time

# ==========================
# 1️⃣ Enhanced Pydantic Modelleri
# ==========================

class ColorPalette(BaseModel):
    primary: str
    secondary: str
    accent: str
    text_primary: str
    text_secondary: str
    background: str
    error: str
    success: str
    warning: str

class Typography(BaseModel):
    primary_font: str
    secondary_font: str
    heading_font: str
    font_sizes: dict
    line_heights: dict

class BrandingGuidelines(BaseModel):
    brand_name: str
    tagline: str
    brand_description: str
    core_values: List[str]
    color_palette: ColorPalette
    typography: Typography
    logo_guidelines: str
    ui_design_principles: str
    accessibility_compliance: str
    brand_voice_tone: str

class PerformanceRequirements(BaseModel):
    response_time_ms: int
    throughput_rps: int
    availability_percentage: float
    scalability_targets: dict

class SecuritySpecs(BaseModel):
    authentication_methods: List[str]
    authorization_framework: str
    data_encryption: dict
    compliance_standards: List[str]
    security_monitoring: List[str]

class TechnicalSpecs(BaseModel):
    technology_stack: dict
    core_components: List[str]
    ai_integration: Optional[dict]
    architecture: str
    third_party_integrations: List[str]
    security: SecuritySpecs
    performance_requirements: PerformanceRequirements
    deployment_strategy: str
    monitoring_logging: dict
    data_architecture: dict

class Feature(BaseModel):
    name: str
    description: str
    priority: str  # P0, P1, P2, P3
    category: str
    user_story: str
    acceptance_criteria: List[str]
    effort_estimation: str  # XS, S, M, L, XL
    dependencies: List[str]
    risks: List[str]

class FeatureList(BaseModel):
    features: List[Feature]
    feature_categories: List[str]
    prioritization_framework: str
    scope_boundaries: dict

class CompetitiveAnalysis(BaseModel):
    direct_competitors: List[str]
    indirect_competitors: List[str]
    competitive_advantages: List[str]
    market_gaps: List[str]
    differentiation_strategy: str

class RiskAssessment(BaseModel):
    technical_risks: List[dict]
    business_risks: List[dict]
    market_risks: List[dict]
    mitigation_strategies: List[str]

class BusinessMetrics(BaseModel):
    success_metrics: List[str]
    kpi_framework: dict
    revenue_projections: dict
    user_acquisition_targets: dict
    retention_goals: dict

class UserPersona(BaseModel):
    name: str
    demographics: dict
    pain_points: List[str]
    goals: List[str]
    behaviors: List[str]

class PRD(BaseModel):
    project_title: str
    executive_summary: str
    objective: str
    problem_statement: str
    target_users: List[UserPersona]
    user_stories: List[str]
    core_features: List[str]
    business_metrics: BusinessMetrics
    competitive_analysis: CompetitiveAnalysis
    risk_assessment: RiskAssessment
    constraints_assumptions: List[str]
    go_to_market_strategy: str
    timeline_milestones: dict
    resource_requirements: dict

class Timeline(BaseModel):
    project_phases: List[dict]
    milestones: List[dict]
    critical_path: List[str]
    dependencies: List[dict]
    estimated_duration: str

class BusinessCase(BaseModel):
    market_opportunity: str
    revenue_model: str
    cost_benefit_analysis: dict
    roi_projections: dict
    market_size_analysis: dict

class UIUXDesign(BaseModel):
    user_flow_diagrams: List[dict]
    wireframe_specifications: List[dict]
    navigation_architecture: dict
    component_library: List[dict]
    accessibility_specifications: dict
    interaction_design: dict
    prototyping_recommendations: dict

class TestPlan(BaseModel):
    testing_strategy: dict
    unit_test_plan: dict
    integration_test_plan: dict
    performance_test_plan: dict
    security_test_plan: dict
    user_acceptance_test_plan: dict
    test_automation_strategy: dict
    test_environments: List[dict]
    quality_gates: List[dict]

class DataArchitecture(BaseModel):
    database_schemas: List[dict]
    api_specifications: dict
    data_models: List[dict]
    api_endpoints: List[dict]
    request_response_examples: List[dict]
    data_flow_diagrams: List[dict]
    data_governance: dict

class DevOpsPipeline(BaseModel):
    ci_cd_stages: List[dict]
    deployment_environments: List[dict]
    infrastructure_as_code: dict
    monitoring_alerting: dict
    rollback_strategies: dict
    security_scanning: dict
    performance_monitoring: dict

class ProductDocs(BaseModel):
    branding: BrandingGuidelines
    technical: TechnicalSpecs
    features: FeatureList
    prd: PRD
    timeline: Timeline
    business_case: BusinessCase
    uiux_design: UIUXDesign
    test_plan: TestPlan
    data_architecture: DataArchitecture
    devops_pipeline: DevOpsPipeline

    def to_dict(self) -> Dict:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

# ==========================
# 2️⃣ API Ayarları
# ==========================

# .env dosyasını yükle (proje kökünden otomatik bulur)
load_dotenv(find_dotenv())

def get_secret_or_env(var_name: str, default_value: Optional[str] = None) -> str:
    """önce ortam/.env, sonra st.secrets, en sonda varsayılan.
    varsayılan da yoksa anlaşılır bir hata ver.
    """
    env_value = os.getenv(var_name)
    if env_value is not None and env_value != "":
        return env_value
    try:
        return st.secrets[var_name]
    except (FileNotFoundError, KeyError):
        if default_value is not None:
            return default_value
        raise RuntimeError(
            f"'{var_name}' değeri .env/ortam ya da st.secrets içinde bulunamadı"
        )

def validate_api_key(api_key: str) -> bool:
    """API anahtarının geçerli format olup olmadığını kontrol eder"""
    if not api_key or len(api_key.strip()) < 10:
        return False
    return api_key.startswith(('sk-', 'org-')) or len(api_key) >= 32

async def test_api_connection(api_key: str) -> tuple[bool, str]:
    """API bağlantısını test eder"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://local.app"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "PRD Creator"),
    }
    test_payload = {
        "model": "openai/gpt-3.5-turbo",  # En ucuz model ile test
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OPENROUTER_URL, headers=headers, 
                                  json=test_payload) as resp:
                if resp.status == 200:
                    return True, "API bağlantısı başarılı"
                elif resp.status == 401:
                    return False, "Geçersiz API anahtarı"
                elif resp.status == 429:
                    return False, "API rate limit aşıldı"
                else:
                    return False, f"API hatası: {resp.status}"
    except asyncio.TimeoutError:
        return False, "API bağlantısı zaman aşımına uğradı"
    except Exception as e:
        return False, f"Bağlantı hatası: {str(e)}"

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Kullanıcı girdisini temizle ve güvenli hale getir"""
    if not text:
        return ""
    
    # HTML escape
    text = html.escape(text.strip())
    
    # Unicode normalizasyonu
    text = unicodedata.normalize('NFKC', text)
    
    # Uzunluk kontrolü
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Zararlı kalıpları filtrele
    dangerous_patterns = [
        r'<script.*?</script>',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'on\w+\s*=',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text

def validate_product_idea(text: str) -> tuple[bool, str]:
    """Validate product idea input"""
    if not text or not text.strip():
        return False, "Product idea cannot be empty"
    
    text = text.strip()
    
    # Minimum length check
    if len(text) < 10:
        return False, "Product idea must be at least 10 characters"
    
    # Maximum length check
    if len(text) > 5000:
        return False, "Product idea cannot be longer than 5000 characters"
    
    # Only space and punctuation check
    if not re.search(r'[a-zA-ZıİşŞğĞüÜöÖçÇ]', text):
        return False, "Product idea must contain meaningful text"
    
    # Repeating character check
    if re.search(r'(.)\1{20,}', text):
        return False, "Invalid repeating character sequence"
    
    return True, ""

def validate_document_quality(docs: ProductDocs) -> dict:
    """Evaluate document quality and scoring"""
    issues = []
    score = 100
    
    # Branding validation
    if len(docs.branding.brand_name) < 3:
        issues.append("Brand name too short")
        score -= 10
    
    if len(docs.branding.tagline) > 50:
        issues.append("Tagline too long (50 characters max recommended)")
        score -= 5
        
    if len(docs.branding.core_values) < 3:
        issues.append("At least 3 core values recommended")
        score -= 10
    
    # Technical validation
    if not docs.technical.core_components:
        issues.append("Core system components missing")
        score -= 15
    
    # Feature validation  
    if not docs.features.features:
        issues.append("No features defined")
        score -= 20
    elif hasattr(docs.features.features[0], 'priority'):
        p0_count = len([f for f in docs.features.features if f.priority == 'P0'])
        if p0_count == 0:
            issues.append("At least 1 P0 (critical) feature required")
            score -= 15
        elif p0_count > 8:
            issues.append("Too many P0 features - MVP scope should be narrowed")
            score -= 10
    
    # PRD validation
    if len(docs.prd.objective) < 50:
        issues.append("Project objective should be more detailed")
        score -= 10
        
    if not hasattr(docs.prd, 'problem_statement') or len(docs.prd.problem_statement) < 30:
        issues.append("Problem definition insufficient")
        score -= 15
    
    # Business case validation
    if hasattr(docs.business_case, 'market_opportunity') and len(docs.business_case.market_opportunity) < 100:
        issues.append("Market opportunity analysis superficial")
        score -= 10
    
    # Timeline validation
    if len(docs.timeline.milestones) < 3:
        issues.append("At least 3 major milestones should be defined")
        score -= 10
    
    # UI/UX Design validation
    if not docs.uiux_design.user_flow_diagrams:
        issues.append("User flow diagrams missing")
        score -= 15
    elif len(docs.uiux_design.user_flow_diagrams) < 2:
        issues.append("At least 2 basic user flows should be defined")
        score -= 10
    
    if not docs.uiux_design.wireframe_specifications:
        issues.append("Wireframe specifications missing")
        score -= 15
    elif len(docs.uiux_design.wireframe_specifications) < 3:
        issues.append("Wireframes for main screens missing")
        score -= 10
    
    if not docs.uiux_design.component_library:
        issues.append("Basic UI component library missing")
        score -= 10
    
    # Test Plan validation
    if not docs.test_plan.test_environments:
        issues.append("Test environments not defined")
        score -= 10
    
    if not docs.test_plan.quality_gates:
        issues.append("Quality control checkpoints missing")
        score -= 10
    
    # Data Architecture validation
    if not docs.data_architecture.database_schemas:
        issues.append("Database schemas not defined")
        score -= 15
    
    if not docs.data_architecture.api_endpoints:
        issues.append("API endpoints missing")
        score -= 10
    
    # DevOps Pipeline validation
    if not docs.devops_pipeline.ci_cd_stages:
        issues.append("CI/CD pipeline stages missing")
        score -= 15
    
    return {
        "score": max(score, 0),
        "issues": issues,
        "total_checks": 18
    }

def get_model_info(model_id: str) -> dict:
    """Model bilgilerini döndür"""
    model_data = {
        # Free Models - Güncel OpenRouter Listesi
        "openai/gpt-oss-20b:free": {
            "name": "GPT-OSS 20B",
            "price": "Ücretsiz",
            "is_free": True,
            "context_length": "32K",
            "provider": "OpenAI"
        },
        "z-ai/glm-4.5-air:free": {
            "name": "GLM 4.5 Air",
            "price": "Ücretsiz", 
            "is_free": True,
            "context_length": "128K",
            "provider": "Z.AI"
        },
        "qwen/qwen3-coder:free": {
            "name": "Qwen3 Coder",
            "price": "Ücretsiz",
            "is_free": True, 
            "context_length": "32K",
            "provider": "Qwen"
        },
        
        # Fast & Economic
        "openai/gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "price": "$0.50/1M tokens",
            "is_free": False,
            "context_length": "16K",
            "provider": "OpenAI"
        },
        "anthropic/claude-3-haiku": {
            "name": "Claude 3 Haiku", 
            "price": "$0.25/1M tokens",
            "is_free": False,
            "context_length": "200K",
            "provider": "Anthropic"
        },
        "google/gemini-flash-1.5": {
            "name": "Gemini 1.5 Flash",
            "price": "$0.075/1M tokens", 
            "is_free": False,
            "context_length": "1M",
            "provider": "Google"
        },
        "mistralai/mistral-7b-instruct": {
            "name": "Mistral 7B",
            "price": "$0.25/1M tokens",
            "is_free": False,
            "context_length": "32K",
            "provider": "Mistral AI"
        },
        
        # High Performance  
        "openai/gpt-4o": {
            "name": "GPT-4o",
            "price": "$2.50/1M tokens",
            "is_free": False,
            "context_length": "128K", 
            "provider": "OpenAI"
        },
        "anthropic/claude-3.5-sonnet": {
            "name": "Claude 3.5 Sonnet",
            "price": "$3.00/1M tokens",
            "is_free": False,
            "context_length": "200K",
            "provider": "Anthropic" 
        },
        "openai/gpt-4-turbo": {
            "name": "GPT-4 Turbo", 
            "price": "$10.00/1M tokens",
            "is_free": False,
            "context_length": "128K",
            "provider": "OpenAI"
        },
        "google/gemini-pro-1.5": {
            "name": "Gemini 1.5 Pro",
            "price": "$2.50/1M tokens",
            "is_free": False,
            "context_length": "1M",
            "provider": "Google"
        },
        "google/gemini-2.5-pro": {
            "name": "Gemini 2.5 Pro",
            "price": "$3.50/1M tokens",
            "is_free": False,
            "context_length": "1M",
            "provider": "Google"
        },
        "qwen/qwen3-30b-a3b-instruct-2507": {
            "name": "Qwen3 30B A3B",
            "price": "$1.50/1M tokens",
            "is_free": False,
            "context_length": "128K",
            "provider": "Qwen"
        },
        
        # Premium & Future
        "openai/gpt-5": {
            "name": "GPT-5", 
            "price": "$25.00/1M tokens",
            "is_free": False,
            "context_length": "200K",
            "provider": "OpenAI"
        },
        "anthropic/claude-3-opus": {
            "name": "Claude 3 Opus",
            "price": "$15.00/1M tokens", 
            "is_free": False,
            "context_length": "200K",
            "provider": "Anthropic"
        },
        "openai/o1-preview": {
            "name": "GPT-o1 Preview",
            "price": "$15.00/1M tokens",
            "is_free": False, 
            "context_length": "128K",
            "provider": "OpenAI"
        },
        "x-ai/grok-beta": {
            "name": "Grok Beta",
            "price": "$5.00/1M tokens",
            "is_free": False,
            "context_length": "25K",
            "provider": "xAI"
        }
    }
    
    return model_data.get(model_id, {
        "name": "Unknown Model",
        "price": "Unknown",
        "is_free": False,
        "context_length": "Unknown",
        "provider": "Unknown"
    })

class RateLimiter:
    """Basit rate limiting sınıfı"""
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def is_allowed(self) -> tuple[bool, int]:
        """İstek izni kontrol et"""
        now = time.time()
        
        # Eski istekleri temizle
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, 0
        else:
            # En eski istek zamanından ne kadar beklemek gerek
            oldest_request = min(self.requests)
            wait_time = int(self.time_window - (now - oldest_request))
            return False, max(wait_time, 1)
    
    def reset(self):
        """Rate limit sayaçını sıfırla"""
        self.requests.clear()

# Global rate limiter (session state ile de yapılabilir)
if 'rate_limiter' not in st.session_state:
    st.session_state.rate_limiter = RateLimiter(max_requests=5, time_window=300)  # 5 dk içinde 5 istek

try:
    OPENROUTER_API_KEY = get_secret_or_env("OPENROUTER_API_KEY")
    if not validate_api_key(OPENROUTER_API_KEY):
        st.error("⚠️ Geçersiz API anahtarı formatı. Lütfen .env dosyasını kontrol edin.")
        st.stop()
except RuntimeError as e:
    st.error(f"⚠️ {str(e)}")
    st.info("💡 Lütfen .env dosyasına OPENROUTER_API_KEY ekleyin veya Streamlit secrets kullanın.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")  # Fallback to GPT-4o

async def call_agent_async(
    session: aiohttp.ClientSession,
    agent_role: str,
    prompt_content: str,
    require_json: bool = False,
    timeout: int = 60,
    model_name: str = None
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter best practice headers
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://local.app"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "PRD Creator"),
    }
    if model_name is None:
        model_name = MODEL_NAME
        
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": agent_role},
            {"role": "user", "content": prompt_content},
        ],
        "temperature": 0.7
    }
    if require_json:
        # bazı modellerde desteklenir; desteklenmiyorsa backend bunu yok sayabilir
        payload["response_format"] = {"type": "json_object"}
    
    try:
        request_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.post(OPENROUTER_URL, headers=headers, 
                              json=payload, timeout=request_timeout) as resp:
            if resp.status == 429:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=429,
                    message="API rate limit aşıldı"
                )
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except asyncio.TimeoutError:
        raise TimeoutError(f"API isteği {timeout} saniye içinde tamamlanamadı")
    except aiohttp.ClientResponseError as e:
        if e.status == 401:
            raise ValueError("API anahtarı geçersiz veya yetkilendirilmemiş")
        elif e.status == 429:
            raise ValueError("API rate limit aşıldı. Lütfen biraz bekleyin")
        elif e.status >= 500:
            raise ConnectionError(f"Server hatası: {e.status}")
        else:
            raise ValueError(f"API hatası: {e.status} - {e.message}")

# ==========================
# 2.1️⃣ Profesyonel Prompt Metni ve Dosya Kaydetme
# ==========================

PROMPT_TEXT_PRO = """Profesyonel AI Promptu – Ürün Geliştirme Öncesi 4 Temel Doküman Üretimi

Rol:
Sen, kıdemli bir ürün yöneticisi, yazılım mimarı ve UI/UX tasarımcısısın. Kullanıcı tarafından sağlanan ürün veya proje fikrine dayanarak, fikir aşamasından ürün geliştirme sürecine kadar ihtiyaç duyulan 4 temel profesyonel dokümanı oluşturmakla görevlisin.

Amaç:
Kullanıcıya, proje başlangıcında net vizyon, teknik gereksinimler, kapsamlı özellik listesi ve marka/tasarım yönergeleri sağlayacak entegre bir doküman seti sunmak.

Talimatlar:
Kullanıcıdan yalnızca bir adet “ürün fikri” veya “proje konusu” alın. Bu bilgi doğrultusunda aşağıdaki 4 dokümanı, birbiriyle uyumlu, uygulanabilir ve profesyonel formatta hazırla.

---

1. Design & Branding Guidelines (Tasarım ve Marka Rehberi)
- Brand Identity (Marka Kimliği)
  - Brand Name: Ürün/marka adı
  - Tagline: Kısa, güçlü, akılda kalıcı slogan
  - Core Values: 3-5 temel değer (ör. inovasyon, güvenlik, erişilebilirlik)
- Color Palette (Renk Paleti)
  - Primary, Secondary, Accent, Text Color (HEX kodları ile)
- Typography (Tipografi)
  - Primary Font, Secondary Font (Bold/Regular varyasyonları ile)
- Logo Guidelines (Logo Kuralları)
  - Kullanım kuralları, minimum boşluk/padding, açık/koyu arka plan uyarlamaları
- UI Design Principles (UI Tasarım Prensipleri)
  - Minimalist, kullanıcı dostu, marka tutarlılığı ve erişilebilirlik standartları (örn. WCAG)

---

2. Technical Specifications Document (Teknik Şartname)
- Technology Stack (Teknoloji Yığını)
  - Frontend, Backend, Database, Cloud Services (sektöre ve ürün tipine uygun olarak)
- Core System Components (Ana Sistem Bileşenleri)
  - Ana modüller ve sorumlulukları
- AI / Automation Integration (varsa)
  - Kullanılacak model türü, barındırma yöntemi, veri işleme pipeline adımları
- Architecture Overview (Mimari)
  - Sistem mimarisi (monolitik, mikroservis), API tasarımı, ölçeklenebilirlik planı
- Third-party Integrations (Üçüncü Taraf Entegrasyonlar)
  - Ödeme sistemleri, analiz araçları, hata raporlama servisleri, sosyal medya API’leri
- Security Specifications (Güvenlik Şartları)
  - Veri şifreleme, erişim kontrolü, güvenlik sertifikaları, GDPR/KVKK uyumu

---

3. Detailed Feature List (Ayrıntılı Özellik Listesi)
- Ana modüller ve her modüldeki alt özellikler
- Kullanıcı onboarding ve kimlik doğrulama yöntemleri
- Yönetim paneli veya kontrol paneli fonksiyonları
- İçerik/medya yönetim özellikleri (varsa)
- Veri işleme, analiz ve raporlama araçları
- İleri düzey kullanıcı fonksiyonları (filtreleme, özelleştirme, entegrasyonlar)

---

4. Product Requirements Document (PRD – Ürün Gereksinim Dokümanı)
- Project Title: Ürün veya proje başlığı
- Objective: Projenin amacı ve çözmek istediği problem
- Target Users: Hedef kullanıcı profilleri (segmentlere göre)
- Core Features: En kritik fonksiyonların özeti
- User Stories: Kullanıcı senaryoları (rol-hedef-fayda formatında)
- Success Metrics: Başarı ölçütleri (ör. kullanıcı sayısı, gelir hedefleri, performans kriterleri)
- Constraints & Assumptions: Teknik, yasal, zaman ve bütçe kısıtlamaları ile varsayımlar

---

Çıktı Formatı:
- Her doküman ayrı ana başlık ve alt maddelerle sunulmalı
- Teknik ve tasarım önerileri uygulanabilir ve net olmalı
- Belgeler arasında kavramsal tutarlılık korunmalı
- Gerektiğinde tablo, madde işaretleri ve kısa açıklamalar kullanılmalı

Kullanıcıdan Alınacak Tek Bilgi:
"Ürün fikrini veya proje konusunu yazınız"
(Örn: “Evcil hayvan sahipleri için yapay zeka destekli sağlık takip mobil uygulaması”)"""

OUTPUT_DIR = Path(__file__).parent / "outputs"

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR

def save_professional_prompt_to_file() -> str:
    """PROMPT_TEXT_PRO metnini proje içi outputs/ klasörüne kaydeder ve yolu döndürür."""
    out_dir = ensure_output_dir()
    file_path = out_dir / "profesyonel_ai_prompt.txt"
    file_path.write_text(PROMPT_TEXT_PRO, encoding="utf-8")
    return str(file_path)

# ==========================
# 3️⃣ Agent Promptları
# ==========================

def get_enhanced_prompts(product_idea: str):
    json_suffix = (
        "Return ONLY valid JSON. No code fences, no explanations. "
        "Output must be UTF-8 encoded single JSON object."
    )
    
    return [
        # 1. Branding Expert
        (
            """You are a senior brand strategist and UI/UX design expert with 15+ years of experience creating brand identities for successful tech products. You understand market positioning, user psychology, and design principles that drive user engagement and brand loyalty.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Create comprehensive branding guidelines that will serve as the foundation for all visual and communication aspects of this product.

ANALYSIS FRAMEWORK:
1. Identify target market segments and their preferences
2. Analyze competitive landscape for differentiation opportunities  
3. Define brand personality and emotional positioning
4. Create cohesive visual identity system
5. Ensure accessibility and inclusive design

DELIVERABLE SCHEMA:
{{
  "brand_name": "string - memorable, distinctive product name",
  "tagline": "string - 3-7 words capturing core value proposition", 
  "brand_description": "string - 2-3 sentences brand essence",
  "core_values": ["string array - 4-6 fundamental brand values"],
  "color_palette": {{
    "primary": "HEX code",
    "secondary": "HEX code", 
    "accent": "HEX code",
    "text_primary": "HEX code",
    "text_secondary": "HEX code", 
    "background": "HEX code",
    "error": "HEX code",
    "success": "HEX code",
    "warning": "HEX code"
  }},
  "typography": {{
    "primary_font": "string - main UI font family",
    "secondary_font": "string - accent/heading font",  
    "heading_font": "string - display/marketing font",
    "font_sizes": {{"h1": "string", "h2": "string", "body": "string", "caption": "string"}},
    "line_heights": {{"heading": "string", "body": "string", "dense": "string"}}
  }},
  "logo_guidelines": "string - logo usage, spacing, do's and don'ts",
  "ui_design_principles": "string - design system principles, component patterns",
  "accessibility_compliance": "string - WCAG compliance level and key considerations", 
  "brand_voice_tone": "string - communication style, personality in content"
}}

Focus on: Market differentiation, user appeal, scalability, and professional execution.
{json_suffix}""",
            BrandingGuidelines,
        ),
        
        # 2. Technical Architect  
        (
            """You are a Principal Software Architect with expertise in scalable systems, cloud architecture, and modern development practices. You've architected solutions for high-growth startups and enterprise products serving millions of users.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Design comprehensive technical architecture that balances performance, scalability, maintainability, and cost-effectiveness.

ANALYSIS FRAMEWORK:
1. Assess functional and non-functional requirements
2. Identify scalability challenges and solutions
3. Design fault-tolerant, secure architecture
4. Plan deployment and operational strategies
5. Consider compliance and regulatory requirements

DELIVERABLE SCHEMA:
{{
  "technology_stack": {{
    "frontend": "string - React/Vue/Angular with reasoning",
    "backend": "string - Node.js/Python/Java/Go with reasoning", 
    "database": "string - SQL/NoSQL choices with reasoning",
    "infrastructure": "string - AWS/GCP/Azure services",
    "cache_layer": "string - Redis/Memcached strategy",
    "message_queue": "string - pub/sub system if needed"
  }},
  "core_components": ["string array - major system modules/services"],
  "ai_integration": {{
    "current_ai_opportunities": "string - immediate AI integration possibilities for this product",
    "recommended_models": ["string array - specific AI models or services that could enhance the product"],
    "implementation_priority": "string - high/medium/low priority for AI features",
    "ai_use_cases": [
      {{
        "use_case": "string - specific AI application",
        "benefit": "string - business value of this AI feature",
        "complexity": "string - implementation difficulty level",
        "timeline": "string - when this could be implemented"
      }}
    ],
    "data_requirements": "string - what data is needed to enable AI features",
    "hosting_strategy": "string - cloud AI services vs self-hosted models",
    "future_ai_roadmap": "string - potential AI enhancements for future versions"
  }},
  "architecture": "string - system architecture pattern (microservices/monolithic/hybrid)",
  "third_party_integrations": ["string array - external APIs/services needed"],
  "security": {{
    "authentication_methods": ["string array - OAuth2, JWT, etc"],
    "authorization_framework": "string - RBAC, ABAC approach",
    "data_encryption": {{"at_rest": "string", "in_transit": "string"}},
    "compliance_standards": ["string array - GDPR, SOC2, etc"],
    "security_monitoring": ["string array - logging, monitoring tools"]
  }},
  "performance_requirements": {{
    "response_time_ms": "number - target API response time",
    "throughput_rps": "number - expected requests per second",
    "availability_percentage": "number - uptime target (99.9)",
    "scalability_targets": {{"users": "string", "data_volume": "string"}}
  }},
  "deployment_strategy": "string - CI/CD, blue-green, canary deployment approach",
  "monitoring_logging": {{"apm": "string", "logs": "string", "metrics": "string", "alerts": "string"}},
  "data_architecture": {{"storage": "string", "backup": "string", "analytics": "string"}}
}}

Consider: Scalability, reliability, security, maintainability, and operational excellence.
{json_suffix}""",
            TechnicalSpecs,
        ),
        
        # 3. Product Features Specialist
        (
            """You are a Senior Product Manager specializing in feature definition and prioritization. You've launched 20+ successful digital products and understand how to balance user needs, business goals, and technical constraints using frameworks like MoSCoW, RICE, and Jobs-to-be-Done.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Define a comprehensive, prioritized feature set that delivers maximum user value while supporting business objectives.

ANALYSIS FRAMEWORK:
1. Map user journeys and identify key pain points
2. Prioritize features using MoSCoW methodology  
3. Define clear acceptance criteria for each feature
4. Assess effort vs impact for development planning
5. Identify feature dependencies and risks

DELIVERABLE SCHEMA:
{{
  "features": [
    {{
      "name": "string - clear, action-oriented feature name",
      "description": "string - detailed feature description and context",
      "priority": "string - P0 (must-have), P1 (should-have), P2 (could-have), P3 (won't-have this release)",
      "category": "string - feature category (core, engagement, monetization, etc)",
      "user_story": "string - As a [user], I want [goal] so that [benefit]",
      "acceptance_criteria": ["string array - specific, testable acceptance criteria"],
      "effort_estimation": "string - XS (1-2 days), S (3-5 days), M (1-2 weeks), L (2-4 weeks), XL (1-2 months)",
      "dependencies": ["string array - other features this depends on"],
      "risks": ["string array - potential risks or blockers"]
    }}
  ],
  "feature_categories": ["string array - major feature groupings"],
  "prioritization_framework": "string - explanation of prioritization rationale and methodology used",
  "scope_boundaries": {{
    "p3_wont_have_features": [
      {{
        "feature_name": "string - feature that won't be included",
        "reason": "string - why this feature is out of scope",
        "future_consideration": "string - when this might be reconsidered",
        "alternative_solution": "string - how users can achieve similar outcomes without this feature"
      }}
    ],
    "scope_creep_risks": ["string array - potential areas where scope might expand"],
    "decision_criteria": "string - how to evaluate future feature requests against current scope"
  }}
}}

Include 15-25 features covering: core functionality, user onboarding, engagement, monetization, admin/management.
Prioritize ruthlessly - P0 features should be sufficient for MVP launch.
{json_suffix}""",
            FeatureList,
        ),
        
        # 4. Product Manager (PRD)
        (
            """You are a VP of Product with extensive experience writing PRDs for successful tech products. You understand how to align stakeholders, define clear success metrics, and create actionable product requirements that engineering and design teams can execute effectively.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Create a comprehensive Product Requirements Document (PRD) that serves as the single source of truth for product development and go-to-market strategy.

ANALYSIS FRAMEWORK:  
1. Define clear problem statement and market opportunity
2. Identify and segment target user personas
3. Analyze competitive landscape and differentiation
4. Set measurable success criteria and KPIs
5. Assess risks and define mitigation strategies

DELIVERABLE SCHEMA:
{{
  "project_title": "string - clear, descriptive product name",
  "executive_summary": "string - 2-3 paragraph high-level overview for executives",
  "objective": "string - primary goal and success definition", 
  "problem_statement": "string - specific problem being solved and why it matters",
  "target_users": [
    {{
      "name": "string - persona name",
      "demographics": {{"age_range": "string", "occupation": "string", "tech_savviness": "string"}},
      "pain_points": ["string array - current challenges they face"],
      "goals": ["string array - what they want to achieve"], 
      "behaviors": ["string array - relevant usage patterns/preferences"]
    }}
  ],
  "user_stories": ["string array - key user scenarios and workflows"],
  "core_features": ["string array - essential features for product success"],
  "business_metrics": {{
    "success_metrics": ["string array - primary success indicators"],
    "kpi_framework": {{"acquisition": "string", "engagement": "string", "retention": "string", "monetization": "string"}},
    "revenue_projections": {{"year_1": "string", "year_2": "string", "year_3": "string"}},
    "user_acquisition_targets": {{"month_1": "string", "month_6": "string", "month_12": "string"}},
    "retention_goals": {{"day_1": "string", "day_7": "string", "day_30": "string"}}
  }},
  "competitive_analysis": {{
    "direct_competitors": ["string array - products solving same problem"],
    "indirect_competitors": ["string array - alternative solutions users might choose"],
    "competitive_advantages": ["string array - our key differentiators"],
    "market_gaps": ["string array - unmet needs we can address"],
    "differentiation_strategy": "string - how we'll position against competition"
  }},
  "risk_assessment": {{
    "technical_risks": [
      {{"risk": "string", "probability": "string", "impact": "string", "mitigation": "string"}}
    ],
    "business_risks": [
      {{"risk": "string", "probability": "string", "impact": "string", "mitigation": "string"}}  
    ],
    "market_risks": [
      {{"risk": "string", "probability": "string", "impact": "string", "mitigation": "string"}}
    ],
    "mitigation_strategies": ["string array - overall risk management approaches"]
  }},
  "constraints_assumptions": ["string array - key limitations and assumptions"],
  "go_to_market_strategy": "string - launch strategy, channels, positioning",
  "timeline_milestones": {{"discovery": "string", "design": "string", "development": "string", "testing": "string", "launch": "string"}},
  "resource_requirements": {{"engineering": "string", "design": "string", "marketing": "string", "budget": "string"}}
}}

Focus on: Clear business case, measurable outcomes, actionable requirements, and stakeholder alignment.
{json_suffix}""",
            PRD,
        ),
        
        # 5. Timeline/Project Manager
        (
            """You are a Senior Technical Program Manager with expertise in agile delivery and product roadmapping. You've successfully managed complex product launches and understand how to balance scope, timeline, and quality while managing dependencies and risks.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Create a realistic project timeline with clear phases, milestones, and dependency mapping for successful product delivery.

ANALYSIS FRAMEWORK:
1. Break down development into logical phases
2. Identify critical path and dependencies  
3. Plan for iterative delivery and feedback loops
4. Account for testing, deployment, and rollback scenarios
5. Include buffer time for unforeseen challenges

DELIVERABLE SCHEMA:
{{
  "project_phases": [
    {{
      "phase_name": "string - phase identifier",
      "duration": "string - estimated time",
      "deliverables": ["string array - key outputs"],
      "success_criteria": ["string array - phase completion criteria"]
    }}
  ],
  "milestones": [
    {{
      "milestone_name": "string - milestone identifier", 
      "target_date": "string - relative timeline (Week X, Month Y)",
      "deliverables": ["string array - what gets delivered"],
      "stakeholders": ["string array - who needs to approve/review"]
    }}
  ],
  "critical_path": ["string array - sequence of critical tasks that determine project duration"],
  "dependencies": [
    {{
      "task": "string - dependent task",
      "depends_on": "string - prerequisite task", 
      "dependency_type": "string - blocking, preference, or informational",
      "risk_level": "string - high, medium, low"
    }}
  ],
  "estimated_duration": "string - total project timeline from start to public launch"
}}

Consider: MVP vs full feature set, parallel workstreams, testing cycles, and market timing.
{json_suffix}""",
            Timeline,
        ),
        
        # 6. Business Analyst
        (
            """You are a Senior Business Analyst and Strategy Consultant with deep expertise in market analysis, business model design, and financial projections. You've helped dozens of startups validate product-market fit and build sustainable business cases.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Develop a comprehensive business case that validates market opportunity and defines the path to profitability.

ANALYSIS FRAMEWORK:
1. Assess total addressable market (TAM) and opportunity size
2. Define business model and revenue streams  
3. Analyze competitive dynamics and market positioning
4. Project financial performance and key business metrics
5. Identify key assumptions and validation requirements

DELIVERABLE SCHEMA:
{{
  "market_opportunity": "string - TAM/SAM analysis, market trends, growth drivers",
  "revenue_model": "string - how the product generates revenue (subscription, transaction, advertising, etc)",
  "cost_benefit_analysis": {{
    "development_costs": "string - estimated development investment",
    "operational_costs": "string - ongoing costs to run the business",
    "customer_acquisition_cost": "string - estimated CAC",
    "break_even_timeline": "string - when product becomes profitable"
  }},
  "roi_projections": {{
    "year_1_revenue": "string - realistic revenue projection",
    "year_2_revenue": "string - growth scenario projection", 
    "year_3_revenue": "string - scale scenario projection",
    "roi_percentage": "string - return on investment calculation"
  }},
  "market_size_analysis": {{
    "tam": "string - total addressable market size",
    "sam": "string - serviceable addressable market",
    "som": "string - serviceable obtainable market", 
    "market_growth_rate": "string - annual market growth projection"
  }}
}}

Focus on: Realistic projections, clear value proposition, and data-driven market insights.
{json_suffix}""",
            BusinessCase,
        ),
        
        # 7. UI/UX Design Specialist
        (
            """You are a Lead UX Designer and Design Systems expert with 10+ years of experience designing user-centered digital products. You specialize in creating intuitive user flows, responsive wireframes, and comprehensive design systems that balance user needs with business goals.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Create comprehensive UI/UX design documentation including wireframes, user flows, and interaction patterns that complement the branding guidelines.

ANALYSIS FRAMEWORK:
1. Map complete user journeys and interaction touchpoints
2. Design responsive wireframe layouts for key screens
3. Define navigation patterns and information architecture
4. Create interactive prototyping specifications
5. Establish design system components and patterns

DELIVERABLE SCHEMA:
{{
  "user_flow_diagrams": [
    {{
      "flow_name": "string - user flow identifier (e.g., User Onboarding, Core Feature Usage)",
      "description": "string - what this flow accomplishes",
      "steps": ["string array - sequential steps in the flow"],
      "decision_points": ["string array - branching logic or user choices"],
      "screen_transitions": ["string array - how users move between screens"]
    }}
  ],
  "wireframe_specifications": [
    {{
      "screen_name": "string - screen identifier (e.g., Landing Page, Dashboard, Settings)",
      "screen_type": "string - mobile, desktop, tablet, or responsive",
      "layout_description": "string - detailed layout structure and component placement",
      "key_elements": ["string array - primary UI components (header, navigation, content areas, CTAs)"],
      "responsive_behavior": "string - how layout adapts across devices",
      "interaction_patterns": ["string array - hover states, animations, micro-interactions"]
    }}
  ],
  "navigation_architecture": {{
    "primary_navigation": ["string array - main menu items"],
    "secondary_navigation": ["string array - sub-menus or contextual nav"],
    "navigation_patterns": "string - navigation style (sidebar, top bar, bottom tabs, etc)",
    "breadcrumb_strategy": "string - how users understand their location"
  }},
  "component_library": [
    {{
      "component_name": "string - component identifier (Button, Card, Form Input)",
      "variants": ["string array - different states/styles"],
      "usage_guidelines": "string - when and how to use this component",
      "responsive_specs": "string - how component behaves across screen sizes"
    }}
  ],
  "accessibility_specifications": {{
    "wcag_compliance_level": "string - AA or AAA compliance target",
    "keyboard_navigation": "string - tab order and keyboard interaction patterns",
    "screen_reader_support": "string - ARIA labels and semantic markup requirements",
    "color_contrast_ratios": "string - minimum contrast requirements for text/backgrounds"
  }},
  "interaction_design": {{
    "animation_principles": "string - timing, easing, and motion design guidelines",
    "feedback_mechanisms": ["string array - loading states, error messages, success confirmations"],
    "gesture_support": ["string array - touch gestures for mobile interfaces"],
    "micro_interactions": ["string array - small delightful interaction details"]
  }},
  "prototyping_recommendations": {{
    "prototype_fidelity": "string - low-fi, mid-fi, or high-fi prototyping approach",
    "testing_scenarios": ["string array - key user scenarios to test with prototypes"],
    "iteration_priorities": ["string array - which areas need most design validation"]
  }}
}}

Focus on: User-centered design, accessibility compliance, mobile-first approach, and seamless user experience across all touchpoints.
{json_suffix}""",
            UIUXDesign,
        ),
        
        # 8. Test Planning Specialist
        (
            """You are a Senior QA Engineer and Test Architect with expertise in comprehensive test planning and quality assurance strategies. You've designed testing frameworks for enterprise applications and understand how to balance test coverage with development velocity.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Create a comprehensive test plan covering all testing phases from unit tests to user acceptance testing, including automation strategies and quality gates.

ANALYSIS FRAMEWORK:
1. Identify critical user journeys and edge cases for testing
2. Design multi-layered testing strategy (unit, integration, system, UAT)
3. Define performance and security testing requirements
4. Plan test automation and CI/CD integration
5. Establish quality gates and acceptance criteria

DELIVERABLE SCHEMA:
{{
  "testing_strategy": {{
    "approach": "string - overall testing philosophy and methodology",
    "test_pyramid": "string - unit/integration/e2e test distribution strategy",
    "risk_based_testing": "string - how to prioritize testing based on risk assessment",
    "shift_left_approach": "string - early testing integration in development"
  }},
  "unit_test_plan": {{
    "coverage_targets": "string - minimum code coverage percentages",
    "testing_frameworks": ["string array - Jest, JUnit, PyTest, etc."],
    "mock_strategies": "string - how to handle dependencies and external services",
    "test_data_management": "string - test data creation and management approach"
  }},
  "integration_test_plan": {{
    "api_testing": "string - REST/GraphQL API testing approach",
    "database_testing": "string - data integrity and migration testing",
    "third_party_integrations": "string - external service testing strategy",
    "contract_testing": "string - consumer-driven contract testing approach"
  }},
  "performance_test_plan": {{
    "load_testing": "string - normal load testing scenarios",
    "stress_testing": "string - peak load and breaking point testing",
    "endurance_testing": "string - sustained load testing approach",
    "performance_criteria": {{"response_time": "string", "throughput": "string", "resource_usage": "string"}}
  }},
  "security_test_plan": {{
    "vulnerability_scanning": "string - automated security scanning tools and approach",
    "penetration_testing": "string - manual security testing strategy",
    "authentication_testing": "string - login, authorization, and access control testing",
    "data_protection_testing": "string - encryption and data privacy testing"
  }},
  "user_acceptance_test_plan": {{
    "test_scenarios": ["string array - key user acceptance test scenarios"],
    "acceptance_criteria": "string - what defines successful UAT completion",
    "user_groups": ["string array - different user types for testing"],
    "feedback_collection": "string - how to gather and incorporate user feedback"
  }},
  "test_automation_strategy": {{
    "automation_scope": "string - which tests to automate vs manual testing",
    "ci_cd_integration": "string - how tests integrate with deployment pipeline",
    "test_maintenance": "string - keeping automated tests up to date",
    "reporting_dashboards": "string - test results visualization and reporting"
  }},
  "test_environments": [
    {{
      "environment_name": "string - dev, staging, production-like",
      "purpose": "string - what this environment is used for",
      "data_requirements": "string - test data needs for this environment",
      "maintenance_schedule": "string - environment refresh and cleanup approach"
    }}
  ],
  "quality_gates": [
    {{
      "gate_name": "string - quality gate identifier",
      "criteria": ["string array - specific pass/fail criteria"],
      "automation_level": "string - automated, manual, or hybrid verification",
      "escalation_process": "string - what happens when gate fails"
    }}
  ]
}}

Focus on: Comprehensive coverage, automation where possible, risk-based prioritization, and continuous quality improvement.
{json_suffix}""",
            TestPlan,
        ),
        
        # 9. Data Architecture Specialist  
        (
            """You are a Senior Data Architect and API Design expert with deep expertise in database design, API architecture, and data governance. You've designed scalable data systems for high-growth products and understand modern data patterns.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Design comprehensive data architecture including database schemas, API contracts, and data governance strategies that support the product's functional and non-functional requirements.

ANALYSIS FRAMEWORK:
1. Model core business entities and their relationships
2. Design API contracts following REST/GraphQL best practices
3. Plan data storage, caching, and retrieval strategies
4. Define data governance and compliance requirements
5. Consider data analytics and reporting needs

DELIVERABLE SCHEMA:
{{
  "database_schemas": [
    {{
      "table_name": "string - database table identifier",
      "purpose": "string - what this table stores",
      "fields": [
        {{
          "field_name": "string - column name",
          "data_type": "string - SQL data type",
          "constraints": "string - primary key, foreign key, not null, etc.",
          "description": "string - field purpose and validation rules"
        }}
      ],
      "relationships": ["string array - foreign key relationships to other tables"],
      "indexes": ["string array - database indexes for performance"]
    }}
  ],
  "api_specifications": {{
    "api_style": "string - REST, GraphQL, or hybrid approach",
    "authentication": "string - JWT, OAuth2, API key strategy",
    "versioning_strategy": "string - URL versioning, header versioning approach",
    "rate_limiting": "string - request throttling and quota management",
    "error_handling": "string - standardized error response format",
    "documentation_standard": "string - OpenAPI, GraphQL schema documentation"
  }},
  "data_models": [
    {{
      "model_name": "string - business entity name",
      "attributes": ["string array - key properties and their types"],
      "validation_rules": ["string array - business logic validation"],
      "lifecycle_events": ["string array - create, update, delete, archive events"]
    }}
  ],
  "api_endpoints": [
    {{
      "endpoint": "string - URL path",
      "method": "string - GET, POST, PUT, DELETE",
      "purpose": "string - what this endpoint does",
      "request_format": "string - expected request body/parameters",
      "response_format": "string - response structure and data types",
      "authentication_required": "boolean - whether auth is needed",
      "rate_limits": "string - specific rate limiting for this endpoint"
    }}
  ],
  "request_response_examples": [
    {{
      "endpoint_name": "string - API endpoint identifier",
      "request_example": "string - JSON example of request",
      "response_example": "string - JSON example of successful response",
      "error_examples": ["string array - common error response examples"]
    }}
  ],
  "data_flow_diagrams": [
    {{
      "flow_name": "string - data flow identifier",
      "description": "string - what data moves where",
      "source_systems": ["string array - where data originates"],
      "processing_steps": ["string array - data transformation steps"],
      "destination_systems": ["string array - where data ends up"],
      "data_quality_checks": ["string array - validation and cleansing steps"]
    }}
  ],
  "data_governance": {{
    "data_privacy": "string - GDPR, CCPA compliance approach",
    "data_retention": "string - how long different data types are stored",
    "data_backup": "string - backup and disaster recovery strategy",
    "data_access_controls": "string - who can access what data and when",
    "audit_logging": "string - tracking data access and modifications"
  }}
}}

Focus on: Scalable design, data integrity, API consistency, and regulatory compliance.
{json_suffix}""",
            DataArchitecture,
        ),
        
        # 10. DevOps Pipeline Specialist
        (
            """You are a Senior DevOps Engineer and Site Reliability expert with extensive experience in CI/CD pipeline design, infrastructure automation, and production operations. You understand how to balance deployment speed with system reliability.""",
            f"""
PRODUCT CONCEPT: "{product_idea}"

Your task: Design comprehensive DevOps pipeline including CI/CD stages, infrastructure automation, monitoring strategies, and operational procedures that ensure reliable and efficient software delivery.

ANALYSIS FRAMEWORK:
1. Design CI/CD pipeline stages from code commit to production deployment
2. Plan infrastructure as code and environment management
3. Define monitoring, alerting, and observability strategies
4. Establish security scanning and compliance automation  
5. Create rollback and disaster recovery procedures

DELIVERABLE SCHEMA:
{{
  "ci_cd_stages": [
    {{
      "stage_name": "string - pipeline stage identifier",
      "trigger": "string - what initiates this stage",
      "actions": ["string array - specific tasks performed in this stage"],
      "tools": ["string array - Jenkins, GitHub Actions, GitLab CI, etc."],
      "success_criteria": "string - when this stage passes",
      "failure_handling": "string - what happens when stage fails",
      "estimated_duration": "string - typical time for stage completion"
    }}
  ],
  "deployment_environments": [
    {{
      "environment_name": "string - dev, staging, production, etc.",
      "purpose": "string - what this environment is used for",
      "deployment_strategy": "string - blue-green, canary, rolling deployment",
      "infrastructure_specs": "string - compute, storage, network requirements",
      "promotion_criteria": "string - requirements to promote to this environment",
      "rollback_procedure": "string - how to revert deployments in this environment"
    }}
  ],
  "infrastructure_as_code": {{
    "iac_tools": ["string array - Terraform, CloudFormation, Pulumi"],
    "configuration_management": "string - Ansible, Chef, Puppet approach",
    "container_orchestration": "string - Kubernetes, Docker Swarm, ECS strategy",
    "infrastructure_testing": "string - how to validate infrastructure changes",
    "cost_optimization": "string - automated cost management and resource optimization"
  }},
  "monitoring_alerting": {{
    "application_monitoring": "string - APM tools and application metrics",
    "infrastructure_monitoring": "string - system metrics, logs, and traces",
    "business_metrics": "string - KPI dashboards and business impact monitoring",
    "alert_escalation": "string - who gets notified when and how",
    "incident_response": "string - procedures for handling production issues",
    "sla_monitoring": "string - tracking and reporting on service level agreements"
  }},
  "rollback_strategies": {{
    "automated_rollback": "string - conditions that trigger automatic rollbacks",
    "manual_rollback": "string - procedures for manual rollback decisions",
    "rollback_testing": "string - how rollback procedures are validated",
    "data_migration_rollback": "string - handling database changes during rollbacks",
    "communication_plan": "string - notifying stakeholders during rollbacks"
  }},
  "security_scanning": {{
    "static_analysis": "string - SAST tools and code quality scanning",
    "dependency_scanning": "string - vulnerability scanning of third-party libraries",
    "container_scanning": "string - Docker image security scanning",
    "infrastructure_scanning": "string - cloud security posture management",
    "compliance_automation": "string - automated compliance checking and reporting"
  }},
  "performance_monitoring": {{
    "response_time_monitoring": "string - API and page load time tracking",
    "resource_utilization": "string - CPU, memory, disk, network monitoring",
    "capacity_planning": "string - predicting and planning for growth",
    "performance_testing_integration": "string - automated performance regression testing",
    "optimization_recommendations": "string - automated suggestions for performance improvements"
  }}
}}

Focus on: Automation, reliability, security, observability, and operational excellence.
{json_suffix}""",
            DevOpsPipeline,
        ),
    ]

# ==========================
# 4️⃣ Markdown Dönüştürücü
# ==========================

def docs_to_markdown(docs: ProductDocs) -> str:
    md = [f"# 📄 Comprehensive Product Documentation\n"]
    
    # Executive Summary
    md.append("## 📋 Executive Summary")
    md.append(f"{docs.prd.executive_summary}\n")
    
    # Branding Guidelines
    md.append("## 🎨 Branding Guidelines")
    md.append(f"**Brand Name:** {docs.branding.brand_name}")
    md.append(f"**Tagline:** {docs.branding.tagline}")
    md.append(f"**Brand Description:** {docs.branding.brand_description}")
    md.append(f"**Core Values:** {', '.join(docs.branding.core_values)}")
    md.append(f"**Brand Voice & Tone:** {docs.branding.brand_voice_tone}")
    
    md.append("### 🎨 Color Palette")
    if hasattr(docs.branding.color_palette, 'primary'):
        md.append(f"- **Primary:** `{docs.branding.color_palette.primary}`")
        md.append(f"- **Secondary:** `{docs.branding.color_palette.secondary}`")
        md.append(f"- **Accent:** `{docs.branding.color_palette.accent}`")
        md.append(f"- **Text Primary:** `{docs.branding.color_palette.text_primary}`")
        md.append(f"- **Background:** `{docs.branding.color_palette.background}`")
    else:
        for k, v in docs.branding.color_palette.items():
            md.append(f"  - {k.capitalize()}: `{v}`")
    
    md.append("### 🔤 Typography")
    if hasattr(docs.branding.typography, 'primary_font'):
        md.append(f"- **Primary Font:** {docs.branding.typography.primary_font}")
        md.append(f"- **Secondary Font:** {docs.branding.typography.secondary_font}")
        md.append(f"- **Heading Font:** {docs.branding.typography.heading_font}")
    else:
        for k, v in docs.branding.typography.items():
            md.append(f"  - {k.capitalize()}: {v}")
    
    md.append(f"**Logo Guidelines:** {docs.branding.logo_guidelines}")
    md.append(f"**UI Design Principles:** {docs.branding.ui_design_principles}")
    md.append(f"**Accessibility Compliance:** {docs.branding.accessibility_compliance}")

    # Technical Specifications
    md.append("\n## 🛠 Technical Specifications")
    md.append("### Technology Stack")
    for k, v in docs.technical.technology_stack.items():
        md.append(f"- **{k.capitalize()}:** {v}")
    
    md.append(f"\n**Core Components:** {', '.join(docs.technical.core_components)}")
    md.append(f"**Architecture:** {docs.technical.architecture}")
    md.append(f"**Deployment Strategy:** {docs.technical.deployment_strategy}")
    
    if docs.technical.ai_integration:
        md.append("### 🤖 AI Integration")
        for k, v in docs.technical.ai_integration.items():
            md.append(f"- **{k.capitalize()}:** {v}")
    
    md.append("### 🔒 Security Specifications")
    if hasattr(docs.technical.security, 'authentication_methods'):
        md.append(f"- **Authentication:** {', '.join(docs.technical.security.authentication_methods)}")
        md.append(f"- **Authorization Framework:** {docs.technical.security.authorization_framework}")
        md.append(f"- **Compliance Standards:** {', '.join(docs.technical.security.compliance_standards)}")
    
    md.append("### ⚡ Performance Requirements")
    if hasattr(docs.technical.performance_requirements, 'response_time_ms'):
        md.append(f"- **Response Time:** {docs.technical.performance_requirements.response_time_ms}ms")
        md.append(f"- **Throughput:** {docs.technical.performance_requirements.throughput_rps} RPS")
        md.append(f"- **Availability:** {docs.technical.performance_requirements.availability_percentage}%")

    # Feature List
    md.append("\n## 📋 Feature Specifications")
    if hasattr(docs.features.features[0], 'name') if docs.features.features else False:
        # New enhanced features
        md.append(f"**Prioritization Framework:** {docs.features.prioritization_framework}")
        md.append("### P0 Features (Must Have)")
        p0_features = [f for f in docs.features.features if f.priority == 'P0']
        for feature in p0_features:
            md.append(f"#### {feature.name}")
            md.append(f"- **Description:** {feature.description}")
            md.append(f"- **User Story:** {feature.user_story}")
            md.append(f"- **Effort:** {feature.effort_estimation}")
            if feature.acceptance_criteria:
                md.append(f"- **Acceptance Criteria:**")
                for criteria in feature.acceptance_criteria:
                    md.append(f"  - {criteria}")
        
        md.append("### P1 Features (Should Have)")
        p1_features = [f for f in docs.features.features if f.priority == 'P1']
        for feature in p1_features[:5]:  # Limit to 5 for brevity
            md.append(f"- **{feature.name}:** {feature.description}")
    else:
        # Fallback for old format
        for f in docs.features.features:
            md.append(f"- {f}")

    # PRD
    md.append("\n## 📑 Product Requirements Document")
    md.append(f"**Project Title:** {docs.prd.project_title}")
    md.append(f"**Objective:** {docs.prd.objective}")
    md.append(f"**Problem Statement:** {docs.prd.problem_statement}")
    
    md.append("### 👥 Target User Personas")
    if hasattr(docs.prd.target_users[0], 'name') if docs.prd.target_users else False:
        for user in docs.prd.target_users:
            md.append(f"#### {user.name}")
            md.append(f"- **Pain Points:** {', '.join(user.pain_points)}")
            md.append(f"- **Goals:** {', '.join(user.goals)}")
    else:
        md.append(f"**Target Users:** {', '.join(docs.prd.target_users)}")
    
    md.append("### 🏆 Competitive Analysis")
    if hasattr(docs.prd.competitive_analysis, 'direct_competitors'):
        md.append(f"- **Direct Competitors:** {', '.join(docs.prd.competitive_analysis.direct_competitors)}")
        md.append(f"- **Competitive Advantages:** {', '.join(docs.prd.competitive_analysis.competitive_advantages)}")
        md.append(f"- **Differentiation Strategy:** {docs.prd.competitive_analysis.differentiation_strategy}")
    
    md.append("### 📊 Business Metrics")
    if hasattr(docs.prd.business_metrics, 'success_metrics'):
        md.append(f"- **Success Metrics:** {', '.join(docs.prd.business_metrics.success_metrics)}")
        md.append("- **Revenue Projections:**")
        for year, revenue in docs.prd.business_metrics.revenue_projections.items():
            md.append(f"  - {year}: {revenue}")
    
    md.append("### ⚠️ Risk Assessment")
    if hasattr(docs.prd.risk_assessment, 'technical_risks'):
        md.append("#### Technical Risks")
        for risk in docs.prd.risk_assessment.technical_risks[:3]:  # Top 3 risks
            md.append(f"- **{risk.get('risk', '')}** (Impact: {risk.get('impact', '')}) - {risk.get('mitigation', '')}")

    # Timeline
    md.append("\n## 📅 Project Timeline")
    md.append(f"**Estimated Duration:** {docs.timeline.estimated_duration}")
    md.append("### Major Milestones")
    for milestone in docs.timeline.milestones:
        md.append(f"- **{milestone.get('milestone_name', '')}** ({milestone.get('target_date', '')}) - {', '.join(milestone.get('deliverables', []))}")

    # Business Case
    md.append("\n## 💼 Business Case")
    md.append(f"**Market Opportunity:** {docs.business_case.market_opportunity}")
    md.append(f"**Revenue Model:** {docs.business_case.revenue_model}")
    md.append("### Market Analysis")
    if hasattr(docs.business_case.market_size_analysis, 'tam'):
        md.append(f"- **TAM:** {docs.business_case.market_size_analysis.tam}")
        md.append(f"- **SAM:** {docs.business_case.market_size_analysis.sam}")
        md.append(f"- **Growth Rate:** {docs.business_case.market_size_analysis.market_growth_rate}")

    # UI/UX Design
    md.append("\n## 🎨 UI/UX Design & Wireframes")
    
    md.append("### 📱 User Flow Diagrams")
    for flow in docs.uiux_design.user_flow_diagrams:
        md.append(f"#### {flow.get('flow_name', 'User Flow')}")
        md.append(f"**Description:** {flow.get('description', '')}")
        if flow.get('steps'):
            md.append("**Flow Steps:**")
            for step in flow['steps']:
                md.append(f"- {step}")
        if flow.get('decision_points'):
            md.append("**Decision Points:**")
            for point in flow['decision_points']:
                md.append(f"- {point}")
    
    md.append("### 📐 Wireframe Specifications")
    for wireframe in docs.uiux_design.wireframe_specifications:
        md.append(f"#### {wireframe.get('screen_name', 'Screen')} ({wireframe.get('screen_type', 'responsive')})")
        md.append(f"**Layout:** {wireframe.get('layout_description', '')}")
        if wireframe.get('key_elements'):
            md.append("**Key Elements:**")
            for element in wireframe['key_elements']:
                md.append(f"- {element}")
        md.append(f"**Responsive Behavior:** {wireframe.get('responsive_behavior', '')}")
    
    md.append("### 🧭 Navigation Architecture")
    nav = docs.uiux_design.navigation_architecture
    if nav.get('primary_navigation'):
        md.append(f"**Primary Navigation:** {', '.join(nav['primary_navigation'])}")
    if nav.get('navigation_patterns'):
        md.append(f"**Navigation Pattern:** {nav['navigation_patterns']}")
    
    md.append("### 🧩 Component Library")
    for component in docs.uiux_design.component_library:
        md.append(f"#### {component.get('component_name', 'Component')}")
        if component.get('variants'):
            md.append(f"**Variants:** {', '.join(component['variants'])}")
        md.append(f"**Usage:** {component.get('usage_guidelines', '')}")
    
    md.append("### ♿ Accessibility Specifications")
    accessibility = docs.uiux_design.accessibility_specifications
    md.append(f"**WCAG Compliance:** {accessibility.get('wcag_compliance_level', 'AA')}")
    md.append(f"**Keyboard Navigation:** {accessibility.get('keyboard_navigation', '')}")
    md.append(f"**Screen Reader Support:** {accessibility.get('screen_reader_support', '')}")

    # Test Plan
    md.append("\n## 🧪 Comprehensive Test Plan")
    
    md.append("### 📋 Testing Strategy")
    strategy = docs.test_plan.testing_strategy
    md.append(f"**Approach:** {strategy.get('approach', '')}")
    md.append(f"**Test Pyramid:** {strategy.get('test_pyramid', '')}")
    
    md.append("### 🔬 Unit Testing")
    unit_plan = docs.test_plan.unit_test_plan
    md.append(f"**Coverage Targets:** {unit_plan.get('coverage_targets', '')}")
    if unit_plan.get('testing_frameworks'):
        md.append(f"**Frameworks:** {', '.join(unit_plan['testing_frameworks'])}")
    
    md.append("### 🔗 Integration Testing")
    integration_plan = docs.test_plan.integration_test_plan
    md.append(f"**API Testing:** {integration_plan.get('api_testing', '')}")
    md.append(f"**Database Testing:** {integration_plan.get('database_testing', '')}")
    
    md.append("### ⚡ Performance Testing")
    perf_plan = docs.test_plan.performance_test_plan
    md.append(f"**Load Testing:** {perf_plan.get('load_testing', '')}")
    md.append(f"**Stress Testing:** {perf_plan.get('stress_testing', '')}")

    # Data Architecture
    md.append("\n## 🗄️ Data Architecture & API Design")
    
    md.append("### 📊 Database Schemas")
    for schema in docs.data_architecture.database_schemas[:3]:  # Show first 3 tables
        md.append(f"#### {schema.get('table_name', 'Table')}")
        md.append(f"**Purpose:** {schema.get('purpose', '')}")
        if schema.get('fields'):
            md.append("**Key Fields:**")
            for field in schema['fields'][:5]:  # Show first 5 fields
                md.append(f"- `{field.get('field_name', '')}` ({field.get('data_type', '')}) - {field.get('description', '')}")
    
    md.append("### 🔌 API Specifications")
    api_specs = docs.data_architecture.api_specifications
    md.append(f"**API Style:** {api_specs.get('api_style', '')}")
    md.append(f"**Authentication:** {api_specs.get('authentication', '')}")
    md.append(f"**Versioning:** {api_specs.get('versioning_strategy', '')}")
    
    md.append("### 🌊 Data Flow")
    for flow in docs.data_architecture.data_flow_diagrams[:2]:  # Show first 2 flows
        md.append(f"#### {flow.get('flow_name', 'Data Flow')}")
        md.append(f"**Description:** {flow.get('description', '')}")
        if flow.get('processing_steps'):
            md.append("**Processing Steps:**")
            for step in flow['processing_steps']:
                md.append(f"- {step}")

    # DevOps Pipeline
    md.append("\n## 🚀 DevOps Pipeline & Operations")
    
    md.append("### 🔄 CI/CD Stages")
    for stage in docs.devops_pipeline.ci_cd_stages[:4]:  # Show first 4 stages
        md.append(f"#### {stage.get('stage_name', 'Stage')}")
        md.append(f"**Trigger:** {stage.get('trigger', '')}")
        if stage.get('actions'):
            md.append("**Actions:**")
            for action in stage['actions']:
                md.append(f"- {action}")
        md.append(f"**Duration:** {stage.get('estimated_duration', '')}")
    
    md.append("### 🏗️ Infrastructure as Code")
    iac = docs.devops_pipeline.infrastructure_as_code
    if iac.get('iac_tools'):
        md.append(f"**IaC Tools:** {', '.join(iac['iac_tools'])}")
    md.append(f"**Container Orchestration:** {iac.get('container_orchestration', '')}")
    md.append(f"**Configuration Management:** {iac.get('configuration_management', '')}")
    
    md.append("### 📊 Monitoring & Alerting")
    monitoring = docs.devops_pipeline.monitoring_alerting
    md.append(f"**Application Monitoring:** {monitoring.get('application_monitoring', '')}")
    md.append(f"**Infrastructure Monitoring:** {monitoring.get('infrastructure_monitoring', '')}")
    md.append(f"**Incident Response:** {monitoring.get('incident_response', '')}")

    return "\n".join(md)

def _model_to_dict(model_obj: BaseModel) -> Dict:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()

def persist_generated_outputs(
    docs: ProductDocs, product_md: str, ide_tasks_md: str, tasks_plan: Optional[Dict] = None
) -> Dict[str, str]:
    """Üretilen çıktıların tamamını outputs/ klasörüne yazar ve dosya yollarını döndürür."""
    out_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths: Dict[str, str] = {}

    # JSON çıktıları
    branding_path = out_dir / f"branding_{timestamp}.json"
    technical_path = out_dir / f"technical_{timestamp}.json"
    features_path = out_dir / f"features_{timestamp}.json"
    prd_path = out_dir / f"prd_{timestamp}.json"
    uiux_path = out_dir / f"uiux_design_{timestamp}.json"
    test_plan_path = out_dir / f"test_plan_{timestamp}.json"
    data_arch_path = out_dir / f"data_architecture_{timestamp}.json"
    devops_path = out_dir / f"devops_pipeline_{timestamp}.json"

    branding_path.write_text(json.dumps(_model_to_dict(docs.branding), ensure_ascii=False, indent=2), encoding="utf-8")
    technical_path.write_text(json.dumps(_model_to_dict(docs.technical), ensure_ascii=False, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(_model_to_dict(docs.features), ensure_ascii=False, indent=2), encoding="utf-8")
    prd_path.write_text(json.dumps(_model_to_dict(docs.prd), ensure_ascii=False, indent=2), encoding="utf-8")
    uiux_path.write_text(json.dumps(_model_to_dict(docs.uiux_design), ensure_ascii=False, indent=2), encoding="utf-8")
    test_plan_path.write_text(json.dumps(_model_to_dict(docs.test_plan), ensure_ascii=False, indent=2), encoding="utf-8")
    data_arch_path.write_text(json.dumps(_model_to_dict(docs.data_architecture), ensure_ascii=False, indent=2), encoding="utf-8")
    devops_path.write_text(json.dumps(_model_to_dict(docs.devops_pipeline), ensure_ascii=False, indent=2), encoding="utf-8")

    paths["branding_json"] = str(branding_path)
    paths["technical_json"] = str(technical_path)
    paths["features_json"] = str(features_path)
    paths["prd_json"] = str(prd_path)
    paths["uiux_json"] = str(uiux_path)
    paths["test_plan_json"] = str(test_plan_path)
    paths["data_architecture_json"] = str(data_arch_path)
    paths["devops_pipeline_json"] = str(devops_path)

    # Markdown çıktıları
    product_md_path = out_dir / f"product_docs_{timestamp}.md"
    tasks_md_path = out_dir / f"dev_tasks_{timestamp}.md"
    product_md_path.write_text(product_md, encoding="utf-8")
    tasks_md_path.write_text(ide_tasks_md, encoding="utf-8")

    paths["product_md"] = str(product_md_path)
    paths["tasks_md"] = str(tasks_md_path)

    # Görev planı (JSON)
    if tasks_plan is not None:
        tasks_json_path = out_dir / f"tasks_plan_{timestamp}.json"
        tasks_json_path.write_text(json.dumps(tasks_plan, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["tasks_plan_json"] = str(tasks_json_path)

    return paths

# ==========================
# 5️⃣ IDE Uyumlu Kanban Task Agent
# ==========================

async def generate_ide_kanban(md_content: str) -> str:
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        role = "You are a senior agile project planner and software architect."
        prompt = f"""
        Aşağıda bir ürünün tüm teknik ve ürün dokümanları var:
        {md_content}

        Görevin:
        - IDE içinde (Cursor, Claude Code, VS Code vb.) doğrudan uygulanabilecek görev listesi hazırla
        - Görevleri "Kanban Board" formatında **4 sütun** halinde ver:
          1. To Do
          2. In Progress
          3. Done
          4. Expected Output
        - Her görev net bir eylem içersin, dosya yolları ve teknolojiler belirtilsin
        - Gerekirse komutları ```bash``` veya ```python``` kod bloğu içinde ekle
        - Öncelik sırasına göre listele
        - Gereksiz açıklama ekleme
        """
        return await call_agent_async(session, role, prompt)

async def generate_structured_tasks_from_docs(docs: ProductDocs, preferred_language: str = "tr") -> Dict:
    """LLM ile 4 dokümana dayalı, hiyerarşik görev planı (JSON) üretir."""
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        role = "You are a senior product/program manager and software architect."
        docs_json = json.dumps(docs.to_dict(), ensure_ascii=False, indent=2)
        prompt = f"""
        Aşağıda bir ürün için 4 temel doküman JSON olarak veriliyor (branding, technical, features, prd):
        {docs_json}

        Görev:
        - Bu dokümanlara dayanarak profesyonel, uygulanabilir, hiyerarşik bir görev planı üret
        - ÇIKTI KESİNLİKLE GEÇERLİ JSON OLSUN. Kod bloğu veya açıklama ekleme
        - Şema:
          {{
            "project": {{
              "title": string,
              "objective": string
            }},
            "tasks": [
              {{
                "id": string,  // benzersiz id, örn: "1", "1.1"
                "title": string,
                "description": string,
                "priority": "high"|"medium"|"low",
                "dependencies": string[],
                "testStrategy": string,
                "acceptanceCriteria": string[],
                "subtasks": [Task]
              }}
            ]
          }}

        Kurallar:
        - Dil: {preferred_language}
        - En az 8-15 ana görev ve her birinde 2-6 alt görev hedefle
        - Görev başlıkları net aksiyon içersin (örn: "Kimlik doğrulama API'sını uygula")
        - Bağımlılıkları id ile belirt (örn: "1", "3.2")
        - Test stratejisi ve kabul kriterleri ekle
        - Sadece JSON üret
        """
        raw = await call_agent_async(session, role, prompt, require_json=True)
        try:
            data = json.loads(raw)
        except Exception:
            # metinden json çıkar
            data = json.loads(_extract_json_from_text(raw))
        return data

# ==========================
# 6️⃣ Async Çalıştırma
# ==========================

def _extract_json_from_text(text: str) -> str:
    """metinden bir json nesnesini güvenle ayıkla ve temizle."""
    # code fence içinde json var mı?
    fence_match = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # dengeli süslü parantez ayıklama
        start = text.find('{')
        if start == -1:
            return text.strip()
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    text = text[start:idx + 1]
                    break
        else:
            text = text[start:].strip()
    
    # JSON temizleme - yaygın formatı bozacak karakterleri düzelt
    text = text.strip()
    
    # Trailing commas temizle (son virgülleri kaldır)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Multiple spaces ve newlines temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Düzensiz quote'ları düzelt
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    return text

def _to_model(model_cls: type[BaseModel], raw_text: str) -> BaseModel:
    """Raw text'i model'e dönüştür - retry logic ile"""
    attempts = [
        lambda x: x,  # Direkt dene
        lambda x: _extract_json_from_text(x),  # JSON extract
        lambda x: _fix_common_json_errors(x),  # Yaygın hataları düzelt
        lambda x: _create_fallback_json(model_cls),  # Son çare - fallback
    ]
    
    last_error = None
    
    for attempt_func in attempts:
        try:
            json_str = attempt_func(raw_text)
            data = json.loads(json_str)
            
            # Pydantic model oluştur
            if hasattr(model_cls, 'model_validate'):
                return model_cls.model_validate(data)  # pydantic v2
            else:
                return model_cls.parse_obj(data)  # pydantic v1
                
        except Exception as e:
            last_error = e
            continue
    
    # Tüm attempts başarısız - hata fırlat
    raise ValueError(f"JSON parsing başarısız. Son hata: {last_error}")

def _fix_common_json_errors(text: str) -> str:
    """JSON'daki yaygın syntax hatalarını düzelt"""
    # İlk önce extract yap
    text = _extract_json_from_text(text)
    
    # Yaygın sorunları düzelt
    fixes = [
        # Trailing commas
        (r',(\s*[}\]])', r'\1'),
        # Missing commas between objects
        (r'}\s*{', r'}, {'),
        # Missing commas between array elements  
        (r']\s*\[', r'], ['),
        # Double quotes in strings fix
        (r'([{,]\s*)"([^"]*)"([^"]*)"(\s*:)', r'\1"\2\3"\4'),
        # Unquoted keys
        (r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
        # Multiple consecutive commas
        (r',+', r','),
        # Newlines and tabs
        (r'[\n\r\t]+', ' '),
        # Multiple spaces
        (r'\s+', ' '),
    ]
    
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def _create_fallback_json(model_cls) -> str:
    """Model için basit fallback JSON oluştur"""
    fallbacks = {
        "BrandingGuidelines": """{
            "brand_name": "Unnamed Product",
            "tagline": "Innovation at its best",
            "brand_description": "A revolutionary product designed for modern users",
            "core_values": ["Innovation", "Quality", "Trust"],
            "color_palette": {
                "primary": "#007bff", "secondary": "#6c757d", "accent": "#17a2b8",
                "text_primary": "#212529", "text_secondary": "#6c757d", 
                "background": "#ffffff", "error": "#dc3545", "success": "#28a745", "warning": "#ffc107"
            },
            "typography": {
                "primary_font": "Inter", "secondary_font": "Roboto", "heading_font": "Montserrat",
                "font_sizes": {"h1": "2.5rem", "h2": "2rem", "body": "1rem", "caption": "0.875rem"},
                "line_heights": {"heading": "1.2", "body": "1.5", "dense": "1.3"}
            },
            "logo_guidelines": "Use minimum 32px spacing around logo. Works on light and dark backgrounds.",
            "ui_design_principles": "Clean, modern design following material design principles.",
            "accessibility_compliance": "WCAG 2.1 AA compliance with proper contrast ratios.",
            "brand_voice_tone": "Professional yet friendly, helpful and trustworthy."
        }""",
        
        "TechnicalSpecs": """{
            "technology_stack": {
                "frontend": "React with TypeScript", "backend": "Node.js with Express", 
                "database": "PostgreSQL", "infrastructure": "AWS", "cache_layer": "Redis"
            },
            "core_components": ["Authentication", "API Gateway", "Database Layer", "User Interface"],
            "ai_integration": {"models": "OpenAI GPT-4", "hosting": "Cloud-based", "data_pipeline": "Real-time"},
            "architecture": "Microservices with API Gateway",
            "third_party_integrations": ["Stripe for payments", "SendGrid for emails", "AWS S3 for storage"],
            "security": {
                "authentication_methods": ["OAuth2", "JWT"], "authorization_framework": "RBAC",
                "data_encryption": {"at_rest": "AES-256", "in_transit": "TLS 1.3"},
                "compliance_standards": ["GDPR", "SOC2"], "security_monitoring": ["CloudWatch", "Sentry"]
            },
            "performance_requirements": {
                "response_time_ms": 200, "throughput_rps": 1000, "availability_percentage": 99.9,
                "scalability_targets": {"users": "100K concurrent", "data_volume": "1TB"}
            },
            "deployment_strategy": "Blue-green deployment with CI/CD",
            "monitoring_logging": {"apm": "New Relic", "logs": "ELK Stack", "metrics": "Prometheus", "alerts": "PagerDuty"},
            "data_architecture": {"storage": "Multi-tier", "backup": "Daily automated", "analytics": "Data warehouse"}
        }""",
        
        "FeatureList": """{
            "features": [
                {
                    "name": "User Authentication", "description": "Secure user login and registration",
                    "priority": "P0", "category": "core", "user_story": "As a user, I want to securely access my account",
                    "acceptance_criteria": ["Login with email/password", "Password reset functionality"],
                    "effort_estimation": "M", "dependencies": [], "risks": ["Security vulnerabilities"]
                },
                {
                    "name": "Dashboard", "description": "Main user interface and navigation",
                    "priority": "P0", "category": "core", "user_story": "As a user, I want to see an overview of my data",
                    "acceptance_criteria": ["Clean interface", "Navigation menu", "Key metrics display"],
                    "effort_estimation": "L", "dependencies": ["User Authentication"], "risks": ["Performance issues"]
                }
            ],
            "feature_categories": ["core", "engagement", "monetization"],
            "prioritization_framework": "MoSCoW methodology used for feature prioritization"
        }""",
        
        "PRD": """{
            "project_title": "New Product Initiative",
            "executive_summary": "This product addresses key market needs with innovative solutions.",
            "objective": "Deliver a user-friendly product that solves core customer problems",
            "problem_statement": "Users currently lack an efficient solution for their daily workflows",
            "target_users": [
                {
                    "name": "Primary User", "demographics": {"age_range": "25-45", "occupation": "Professional", "tech_savviness": "High"},
                    "pain_points": ["Time consuming tasks", "Complex workflows"], "goals": ["Efficiency", "Simplicity"],
                    "behaviors": ["Mobile-first", "Values automation"]
                }
            ],
            "user_stories": ["As a user, I want to complete tasks quickly and efficiently"],
            "core_features": ["Authentication", "Dashboard", "Core functionality"],
            "business_metrics": {
                "success_metrics": ["User engagement", "Retention rate", "Customer satisfaction"],
                "kpi_framework": {"acquisition": "Monthly signups", "engagement": "Daily active users", "retention": "30-day retention", "monetization": "Revenue per user"},
                "revenue_projections": {"year_1": "$100K", "year_2": "$500K", "year_3": "$1M"},
                "user_acquisition_targets": {"month_1": "100 users", "month_6": "1K users", "month_12": "10K users"},
                "retention_goals": {"day_1": "80%", "day_7": "60%", "day_30": "40%"}
            },
            "competitive_analysis": {
                "direct_competitors": ["Competitor A", "Competitor B"], "indirect_competitors": ["Alternative Solution"],
                "competitive_advantages": ["Better UX", "Lower cost", "Faster performance"],
                "market_gaps": ["Unmet user needs"], "differentiation_strategy": "Focus on user experience and affordability"
            },
            "risk_assessment": {
                "technical_risks": [{"risk": "Scalability issues", "probability": "Medium", "impact": "High", "mitigation": "Load testing and optimization"}],
                "business_risks": [{"risk": "Market competition", "probability": "High", "impact": "Medium", "mitigation": "Unique value proposition"}],
                "market_risks": [{"risk": "Economic downturn", "probability": "Low", "impact": "High", "mitigation": "Diversified revenue streams"}],
                "mitigation_strategies": ["Regular risk assessment", "Agile development", "Market research"]
            },
            "constraints_assumptions": ["Budget constraints", "Timeline limitations", "Technical constraints"],
            "go_to_market_strategy": "Digital marketing campaign with focus on early adopters",
            "timeline_milestones": {"discovery": "Month 1", "design": "Month 2", "development": "Month 3-5", "testing": "Month 6", "launch": "Month 7"},
            "resource_requirements": {"engineering": "3 developers", "design": "1 designer", "marketing": "1 marketer", "budget": "$100K"}
        }""",
        
        "Timeline": """{
            "project_phases": [
                {"phase_name": "Discovery", "duration": "4 weeks", "deliverables": ["Requirements document", "User research"], "success_criteria": ["Stakeholder approval"]},
                {"phase_name": "Design", "duration": "3 weeks", "deliverables": ["UI/UX designs", "Prototypes"], "success_criteria": ["Design approval"]},
                {"phase_name": "Development", "duration": "8 weeks", "deliverables": ["MVP", "Testing"], "success_criteria": ["Feature complete"]},
                {"phase_name": "Launch", "duration": "2 weeks", "deliverables": ["Production deployment"], "success_criteria": ["Go-live"]}
            ],
            "milestones": [
                {"milestone_name": "Requirements Complete", "target_date": "Week 4", "deliverables": ["PRD", "Technical specs"], "stakeholders": ["PM", "Engineering"]},
                {"milestone_name": "MVP Complete", "target_date": "Week 15", "deliverables": ["Working product"], "stakeholders": ["All teams"]}
            ],
            "critical_path": ["Requirements", "Design", "Core development", "Testing", "Launch"],
            "dependencies": [
                {"task": "Development", "depends_on": "Design approval", "dependency_type": "blocking", "risk_level": "medium"},
                {"task": "Testing", "depends_on": "Feature complete", "dependency_type": "blocking", "risk_level": "high"}
            ],
            "estimated_duration": "17 weeks from project start to public launch"
        }""",
        
        "BusinessCase": """{
            "market_opportunity": "Large addressable market with growing demand for digital solutions",
            "revenue_model": "Subscription-based with tiered pricing",
            "cost_benefit_analysis": {
                "development_costs": "$150K", "operational_costs": "$50K/year",
                "customer_acquisition_cost": "$25", "break_even_timeline": "18 months"
            },
            "roi_projections": {
                "year_1_revenue": "$200K", "year_2_revenue": "$800K", "year_3_revenue": "$2M",
                "roi_percentage": "300% by year 3"
            },
            "market_size_analysis": {
                "tam": "$10B total addressable market", "sam": "$1B serviceable addressable market",
                "som": "$100M serviceable obtainable market", "market_growth_rate": "15% annually"
            }
        }"""
    }
    
    class_name = model_cls.__name__
    return fallbacks.get(class_name, '{"error": "Fallback not available"}')

async def generate_all(product_idea: str, model_name: str = None, progress_callback=None):
    if model_name is None:
        model_name = MODEL_NAME
        
    prompts = get_enhanced_prompts(product_idea)
    agent_names = [
        "🎨 Brand Strategist", "🏗️ Principal Architect", "📋 Senior PM", 
        "📑 VP Product", "📅 Program Manager", "💼 Business Analyst",
        "🎨 UX/UI Designer", "🧪 QA Test Architect", "🗄️ Data Architect", 
        "🚀 DevOps Engineer"
    ]
    
    timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 dk total, 30s bağlantı
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = []
        
        # Run each agent sequentially and update progress
        for i, (role, prompt, _) in enumerate(prompts):
            if progress_callback:
                progress = 20 + (i * 50 // len(prompts))  # 20-70 progress range
                progress_callback(progress, f"🤖 {agent_names[i]} working...")
            
            result = await call_agent_async(session, role, prompt, require_json=True, timeout=120, model_name=model_name)
            results.append(result)

    if progress_callback:
        progress_callback(70, "🔄 Processing AI responses...")
    
    branding = _to_model(prompts[0][2], results[0])
    technical = _to_model(prompts[1][2], results[1])
    features = _to_model(prompts[2][2], results[2])
    prd = _to_model(prompts[3][2], results[3])
    timeline = _to_model(prompts[4][2], results[4])
    business_case = _to_model(prompts[5][2], results[5])
    uiux_design = _to_model(prompts[6][2], results[6])
    test_plan = _to_model(prompts[7][2], results[7])
    data_architecture = _to_model(prompts[8][2], results[8])
    devops_pipeline = _to_model(prompts[9][2], results[9])
    
    docs = ProductDocs(
        branding=branding, 
        technical=technical, 
        features=features, 
        prd=prd,
        timeline=timeline,
        business_case=business_case,
        uiux_design=uiux_design,
        test_plan=test_plan,
        data_architecture=data_architecture,
        devops_pipeline=devops_pipeline
    )

    if progress_callback:
        progress_callback(80, "📝 Generating documents...")
    
    md_content = docs_to_markdown(docs)
    
    if progress_callback:
        progress_callback(90, "🗂️ Preparing IDE tasks...")
    
    ide_tasks_md = await generate_ide_kanban(md_content)

    # Additionally, generate hierarchical task plan from docs
    try:
        tasks_plan = await generate_structured_tasks_from_docs(docs)
    except Exception:
        tasks_plan = None

    if progress_callback:
        progress_callback(100, "✅ Completed!")
    
    return docs, md_content, ide_tasks_md, tasks_plan

# ==========================
# 7️⃣ Streamlit UI
# ==========================

st.set_page_config(
    page_title="PRD Creator - AI Product Documentation Generator", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear outputs folder on startup
def clear_outputs_folder():
    """Clear all files in the outputs folder on application startup."""
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for file in outputs_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception:
                    pass  # Ignore file deletion errors

async def generate_initial_document(product_idea: str, model_name: str) -> str:
    """Generate initial.md document based on CLAUDE.md guidelines."""
    
    # Read CLAUDE.md guidelines
    claude_md_path = Path("Examples/CLAUDE.md")
    claude_guidelines = ""
    if claude_md_path.exists():
        with open(claude_md_path, "r", encoding="utf-8") as f:
            claude_guidelines = f.read()
    
    # Create initial document prompt
    initial_prompt = f"""You are an expert software architect creating an initial project specification document based on CLAUDE.md guidelines.

CLAUDE.MD GUIDELINES:
{claude_guidelines}

USER'S PRODUCT IDEA:
{product_idea}

Please create an initial.md document that follows the CLAUDE.md guidelines. The document should include:

## FEATURE:
- Brief description of the main features and functionality
- Core technology requirements
- Key integrations needed

## EXAMPLES:
- Reference to any relevant examples or patterns to follow
- Best practices to implement

## DOCUMENTATION:
- Links to relevant documentation
- API references needed

## OTHER CONSIDERATIONS:
- Setup requirements
- Environment considerations
- Testing approach
- Any specific requirements

Format as a clear markdown document that follows the CLAUDE.md structure and principles. Be concise but include all necessary information for development.

Return only the markdown content, no additional text."""

    timeout = aiohttp.ClientTimeout(total=120, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await call_agent_async(
            session, 
            "Initial Document Agent", 
            initial_prompt, 
            require_json=False, 
            timeout=60, 
            model_name=model_name
        )
    
    return result

async def generate_comprehensive_prp(initial_md: str, product_idea: str, model_name: str, app_name: str) -> str:
    """Generate comprehensive PRP document using prp_base.md template and EXAMPLE_multi_agent_prp.md as reference."""
    
    # Read template files
    prp_base_path = Path("Examples/prp_base.md")
    example_prp_path = Path("Examples/EXAMPLE_multi_agent_prp.md")
    
    prp_base = ""
    example_prp = ""
    
    if prp_base_path.exists():
        with open(prp_base_path, "r", encoding="utf-8") as f:
            prp_base = f.read()
    
    if example_prp_path.exists():
        with open(example_prp_path, "r", encoding="utf-8") as f:
            example_prp = f.read()
    
    # Create comprehensive PRP prompt
    prp_prompt = f"""You are an expert technical project manager creating a comprehensive Project Requirements and Procedures (PRP) document.

PRP BASE TEMPLATE:
{prp_base}

EXAMPLE PRP DOCUMENT (for reference):
{example_prp}

INITIAL PROJECT SPECIFICATION:
{initial_md}

ORIGINAL PRODUCT IDEA:
{product_idea}

APPLICATION NAME: {app_name}

Please create a comprehensive PRP document for "{app_name}" using the prp_base.md template structure and following the patterns shown in EXAMPLE_multi_agent_prp.md.

The PRP should include:

1. **Purpose & Core Principles** - Based on the initial.md analysis
2. **Goal** - Clear, specific end state for {app_name}  
3. **Why** - Business value, integration points, problems solved
4. **What** - User-visible behavior and technical requirements
5. **Success Criteria** - Measurable outcomes
6. **All Needed Context** - Documentation, references, codebase context
7. **Implementation Blueprint** - Data models, task breakdown, integration points
8. **Validation Loop** - Testing strategy, quality gates
9. **Anti-Patterns to Avoid**

Focus on making this actionable for AI agents to implement. Include specific file structures, code patterns, and validation steps.

Return a comprehensive markdown document following the prp_base.md structure."""

    timeout = aiohttp.ClientTimeout(total=180, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await call_agent_async(
            session, 
            "PRP Analysis Agent", 
            prp_prompt, 
            require_json=False, 
            timeout=120, 
            model_name=model_name
        )
    
    return result

def save_new_workflow_outputs(initial_md: str, comprehensive_prp: str, app_name: str):
    """Save only initial.md and appname_prp.md files to outputs folder."""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Save initial.md
    initial_path = outputs_dir / "initial.md"
    with open(initial_path, "w", encoding="utf-8") as f:
        f.write(initial_md)
    
    # Save comprehensive PRP
    prp_filename = f"{app_name.replace(' ', '_')}_prp.md"
    prp_path = outputs_dir / prp_filename
    with open(prp_path, "w", encoding="utf-8") as f:
        f.write(comprehensive_prp)
    
    return {
        "initial_md": str(initial_path),
        "comprehensive_prp": str(prp_path)
    }

async def generate_new_workflow(product_idea: str, model_name: str = None, progress_callback=None):
    """New 2-step workflow: Generate initial.md then comprehensive PRP document."""
    if model_name is None:
        model_name = MODEL_NAME
    
    # Extract app name from product idea
    app_name = product_idea.split('.')[0].split(',')[0].strip()
    if len(app_name) > 50:
        app_name = app_name[:50] + "..."
    app_name = re.sub(r'[^\w\s-]', '', app_name).strip()
    if not app_name:
        app_name = "Product"
    
    # Step 1: Generate initial.md
    if progress_callback:
        progress_callback(20, "📋 Initial Document Agent analyzing requirements...")
    
    initial_md = await generate_initial_document(product_idea, model_name)
    
    # Step 2: Generate comprehensive PRP
    if progress_callback:
        progress_callback(60, "🔬 PRP Analysis Agent creating comprehensive specification...")
    
    comprehensive_prp = await generate_comprehensive_prp(initial_md, product_idea, model_name, app_name)
    
    if progress_callback:
        progress_callback(100, "✅ Completed!")
    
    return initial_md, comprehensive_prp, app_name

# Initialize session state and clear outputs
if 'app_initialized' not in st.session_state:
    clear_outputs_folder()
    st.session_state.app_initialized = True

# Sidebar info
with st.sidebar:
    st.header("ℹ️ PRD Creator")
    st.write("🤖 **2-Step AI Workflow** for project specifications")
    
    with st.expander("📋 AI Workflow"):
        st.write("""
        **Step 1:** 📋 Initial Document Agent
        • Analyzes requirements using CLAUDE.md guidelines
        • Creates initial.md specification
        
        **Step 2:** 🔬 PRP Analysis Agent  
        • Reviews initial specification
        • Generates comprehensive PRP document
        • Uses prp_base.md template structure
        """)
    
    st.divider()
    st.subheader("🤖 Models")
    st.write("""
    **🆓 Free:** GPT-OSS, GLM 4.5, Qwen3  
    **⚡ Performance:** GPT-4o, Claude 3.5 Sonnet
    **🚀 Premium:** GPT-5, Claude Opus
    """)
    
    if 'rate_limiter' in st.session_state:
        remaining = st.session_state.rate_limiter.max_requests - len(st.session_state.rate_limiter.requests)
        st.metric("Remaining Requests", f"{max(0, remaining)}/5")

st.title("🚀 AI Powered Project Specification Creator")
st.markdown("""
### From Product Ideas to Comprehensive PRP Documents  
Transform your product ideas into structured project specifications using a 2-step AI workflow that creates initial requirements and comprehensive Project Requirements & Procedures (PRP) documents.
""")

# API connection test in sidebar
with st.sidebar:
    st.divider()
    if st.button("🔍 Test API Connection"):
        with st.spinner("Testing..."):
            try:
                is_connected, message = asyncio.run(test_api_connection(OPENROUTER_API_KEY))
                if is_connected:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Test error: {e}")

product_idea = st.text_area(
    "💡 Describe your product idea in detail", 
    placeholder="Example: AI-powered health tracking app for pets. Tracks veterinary checkups, creates vaccination schedules, and analyzes health data to alert owners.",
    height=120,
    help="The more detailed you are, the better documents will be generated. Minimum 10 characters required."
)

# Character count and preview
if product_idea:
    char_count = len(product_idea)
    if char_count < 10:
        st.warning(f"⚠️ Too short ({char_count}/10 minimum characters)")
    elif char_count > 5000:
        st.error(f"❌ Too long ({char_count}/5000 maximum characters)")
    else:
        st.info(f"✅ Appropriate length ({char_count} characters)")
    
    if char_count >= 10:
        with st.expander("🔍 Input Preview"):
            sanitized_preview = sanitize_input(product_idea, max_length=200)
            st.write(f"**Sanitized input:** {sanitized_preview}")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📝 Save Professional Prompt Text"):
        try:
            saved_path = save_professional_prompt_to_file()
            st.success(f"Professional prompt saved: {saved_path}")
        except Exception as e:
            st.error(f"Could not save prompt: {e}")

with col2:
    # Simplified model selection - more secure
    st.write("**🤖 AI Model Selection**")
    
    # Flat model list - without categories
    all_models = [
        # Free models
        ("openai/gpt-oss-20b:free", "🆓 GPT-OSS 20B (Free)"),
        ("z-ai/glm-4.5-air:free", "🆓 GLM 4.5 Air (Free)"), 
        ("qwen/qwen3-coder:free", "🆓 Qwen3 Coder (Free)"),
        # Economic models
        ("openai/gpt-3.5-turbo", "⚡ GPT-3.5 Turbo ($0.50/1M)"),
        ("anthropic/claude-3-haiku", "⚡ Claude 3 Haiku ($0.25/1M)"),
        ("google/gemini-flash-1.5", "⚡ Gemini 1.5 Flash ($0.075/1M)"),
        # Performance models
        ("openai/gpt-4o", "🏆 GPT-4o ($2.50/1M) - Recommended"),
        ("anthropic/claude-3.5-sonnet", "🏆 Claude 3.5 Sonnet ($3.00/1M)"),
        ("google/gemini-2.5-pro", "🏆 Gemini 2.5 Pro ($3.50/1M)"),
        ("qwen/qwen3-30b-a3b-instruct-2507", "🏆 Qwen3 30B ($1.50/1M)"),
        # Premium models
        ("openai/gpt-5", "🚀 GPT-5 ($25.00/1M) - Most Advanced"),
        ("anthropic/claude-3-opus", "🚀 Claude 3 Opus ($15.00/1M)"),
    ]
    
    # Find default (GPT-4o) 
    default_index = 6  # GPT-4o position in list
    
    model_choice = st.selectbox(
        "Select Model",
        all_models,
        format_func=lambda x: x[1],  # Display name
        index=default_index,
        help="🆓=Free, ⚡=Economic, 🏆=Performance, 🚀=Premium"
    )
    
    selected_model = model_choice[0]  # Get model ID
    
    # Show model info
    try:
        model_info = get_model_info(selected_model)
        if model_info and model_info.get("is_free"):
            st.success(f"✅ Completely Free!")
        elif model_info and model_info.get("price"):
            st.info(f"💰 Cost: {model_info['price']}")
        
        if model_info and model_info.get("context_length"):
            st.caption(f"📝 Context: {model_info['context_length']} tokens")
    except Exception:
        st.caption(f"Model: {selected_model}")
    
    save_to_disk = st.checkbox("Save outputs to outputs/ folder in project", value=True)

if st.button("🚀 Generate Project Specifications"):
    # Rate limiting check
    allowed, wait_time = st.session_state.rate_limiter.is_allowed()
    if not allowed:
        st.error(f"⏱️ Too many requests! Try again in {wait_time} seconds.")
        st.info("This protection is to optimize API costs.")
    else:
        # Input validation
        is_valid, error_message = validate_product_idea(product_idea)
        if not is_valid:
            st.error(f"⚠️ {error_message}")
        else:
            # Input sanitization
            sanitized_idea = sanitize_input(product_idea)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Progress callback function
                def update_progress(progress_value, status_message):
                    progress_bar.progress(progress_value)
                    status_text.text(status_message)
                
                # Initial progress
                update_progress(10, "🚀 Initializing process...")
                
                initial_md, comprehensive_prp, app_name = asyncio.run(
                    generate_new_workflow(sanitized_idea, selected_model, progress_callback=update_progress)
                )
                
                status_text.text("✅ Project specification documents successfully generated!")
                progress_bar.progress(100)
                
                # Clear progress bar and status
                import time
                time.sleep(1)  # Brief display
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"🏆 Successfully created project specifications for: {app_name}")

                st.subheader("📋 Initial Project Specification")
                st.markdown(initial_md)
                st.download_button("📥 Download Initial Specification", initial_md, "initial.md", "text/markdown")

                st.subheader("📑 Comprehensive PRP Document")
                st.markdown(comprehensive_prp)
                st.download_button("📥 Download Comprehensive PRP", comprehensive_prp, f"{app_name}_prp.md", "text/markdown")

                if save_to_disk:
                    try:
                        paths = save_new_workflow_outputs(initial_md, comprehensive_prp, app_name)
                        st.success("✅ Documents saved to outputs/ folder.")
                        st.code(json.dumps(paths, ensure_ascii=False, indent=2), language="json")
                    except Exception as save_error:
                        st.warning(f"⚠️ File save error: {save_error}")
                        st.info("Documents are only displayed in browser, you can use download buttons.")

            except ValueError as ve:
                if "API anahtarı" in str(ve) or "403" in str(ve):
                    st.error("🔑 API Key Issue Detected")
                    st.info("""
                    **Solution Suggestions:**
                    1. Ensure you have credits in your OpenRouter account
                    2. Check that your API key is correct
                    3. Verify you have access permission for the selected model
                    4. Try a cheaper model (GPT-3.5-turbo)
                    """)
                elif "rate limit" in str(ve).lower():
                    st.error("⏱️ API rate limit exceeded. Please wait a few minutes and try again.")
                else:
                    st.error(f"⚠️ API Error: {ve}")
                    if "403" in str(ve):
                        st.warning("💡 403 Forbidden: Could be an API key or model access permission issue.")
            
            except TimeoutError as te:
                st.error("⏳ Operation timed out. API response times may be long, please try again.")
                st.info("💡 Tip: Try a shorter product idea description.")
            
            except ConnectionError as ce:
                st.error("🌐 Connection problem: Check your internet connection.")
                st.info(f"Detail: {ce}")
            
            except json.JSONDecodeError as je:
                st.error("📝 AI model returned invalid JSON - automatic correction system engaged.")
                st.info("💡 This usually happens with free models. Try a premium model or try again.")
                if st.checkbox("Show JSON Error Details"):
                    st.code(str(je))
                    st.caption("This error is caused by the AI model giving a response that doesn't conform to JSON format rules.")
            
            except Exception as e:
                st.error("❌ An unexpected error occurred.")
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                with st.expander("🔍 Error Details"):
                    st.code(json.dumps(error_details, ensure_ascii=False, indent=2), language="json")
                    st.info("If you want to report this error, you can copy the details above.")
