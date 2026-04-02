# AI Thực Chiến — Chương Trình Đào Tạo

**Tổ chức:** VinUniversity  
---

## Giới Thiệu

**AI Thực Chiến** là chương trình đào tạo thực hành chuyên sâu dành cho kỹ sư và lập trình viên muốn xây dựng sản phẩm AI thực tế. Chương trình bao gồm toàn bộ vòng đời phát triển AI: từ gọi API đến triển khai production, từ RAG pipeline đến fine-tuning, từ agent đơn giản đến hệ thống multi-agent phức tạp.

---

## Cấu Trúc Chương Trình

| Ngày | Chủ Đề | Vibe Coding + LAB | Kỹ Năng Chính |
|------|--------|-------------------|---------------|
| 01 | AI & LLM Foundation | Vibe Coding 1: Setup & Foundation + LAB 1: LLM API Exploration | Cấu trúc Transformer, Tokenization, Gọi API đầu tiên |
| 02 | Xác định Bài toán Kinh doanh cho AI | Vibe Coding 2: AI-powered Business Analysis + LAB 2: Problem Statement & Use Case Mapping | AI Readiness, ROI, Problem Statement |
| 03 | Design Pattern ReAct | Vibe Coding 3: Agent Loop Prototyping + LAB 3: Chatbot vs Agent: Hands-on Comparison | Kiến trúc Agent, Agent Loop, Thought-Action-Obs |
| 04 | Prompt Engineering & Tool Calling | Vibe Coding 4: Tool Development with AI + LAB 4: Build First AI Agent - ReAct + Tools | System prompt, Function calling, LangGraph |
| 05 | AI Product Thinking & Requirements | Vibe Coding 5: PRD Generation & Risk Analysis + LAB 5: PRD Writing & Risks Assessment | PRD cho AI, Risk Assessment |
| 06 | AI Product & Project Management | Vibe Coding 6: Business Document Generation + LAB 6: Final PRD and ROI Analysis | Agile cho AI, MVP, Low-code tools |
| 07 | Data Foundations | Vibe Coding 7: Data Processing Pipeline + LAB 7: Vector Store Integration | Embedding, Chunking, Vector Store — ChromaDB/FAISS |
| 08 | RAG Pipeline — Truy Xuất & Sinh Câu Trả Lời | Vibe Coding 8: RAG Pipeline Building + LAB 8: Full RAG Pipeline | Hybrid search, Re-ranking, RAG evaluation |
| 09 | Multi-Agent & MCP/A2A | Vibe Coding 9: Multi-agent Architecture + LAB 9: Supervisor Pattern + MCP Integration | Supervisor pattern, Agent-to-Agent communication |
| 10 | Data Pipeline & Data Observability | Vibe Coding 10: ETL Automation + LAB 10: Data Quality Impact on Agent | Garbage in → garbage out — fix thế nào? |
| 11 | Guardrails, HITL & Responsible AI | Vibe Coding 11: Security Testing and Protection + LAB 11: Attack, Defend, and Human Control | Làm sao để ứng dụng agent an toàn? |
| 12 | Hạ Tầng Cloud & Deployment | Vibe Coding 12: Containerization & CI/CD + LAB 12: Docker → Cloud → API Live | Agent ra khỏi laptop — deploy thế nào? |
| 13 | Monitoring & LLMOps | Vibe Coding 13: Observability Setup + LAB 13: Production Monitoring & Prompt Versioning | Agent chạy rồi — sao biết nó ổn? |
| 14 | Evaluation Pipeline & Failure Analysis | Vibe Coding 14: Automated Eval Pipeline + LAB 14: Benchmark, Evaluate, and Improve | Đo agent tốt hay không tốt bằng gì? |
| 15 | Triển Khai Agent Thực Tế, Chi Phí Vận Hành & Định Hướng Chuyên Sâu | Vibe Coding 15: Cost Optimization & Portfolio + LAB 15: Cost Analysis & Final Presentation | Cost optimization, portfolio review |

---

## Cấu Trúc Thư Mục

```
AI_thucchien/
├── day_01_llm_api_foundation/
│   ├── template.py          # Bài tập — điền TODO
│   ├── tests/               # Kiểm thử tự động
│   ├── exercises.md         # Câu hỏi và phần phản ánh
│   ├── README.md            # Hướng dẫn ngày học
│   └── solution/            # Nộp bài tại đây
├── day_02_prompt_engineering/
│   └── ...
└── requirements.txt
```

---

## Bắt Đầu

### Yêu Cầu
- Python 3.10+
- OpenAI API key

### Cài Đặt

```bash
git clone <repo-url>
cd AI_thucchien
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

### Quy Trình Mỗi Ngày

```bash
cd day_XX_<ten_chu_de>

# Đọc hướng dẫn
cat README.md

# Triển khai các TODO trong template
# (mở template.py trong editor)

# Kiểm tra tiến độ
pytest tests/ -v

# Chạy thử thủ công
python template.py

# Nộp bài
cp template.py solution/solution.py
```

---

## Kiểm Thử

Mỗi ngày đều có bộ kiểm thử tự động dùng mock — **không cần API key thật để chạy tests**.

```bash
pytest tests/ -v          # Chạy tất cả tests
pytest tests/ -k "test_X" # Chạy test cụ thể
```

---

## Tiêu Chí Đánh Giá

| Hạng Mục | Tỉ Lệ |
|----------|-------|
| Tests pass (`pytest`) | 60% |
| Chất lượng triển khai | 25% |
| Bài tập & phản ánh | 15% |

---

## Liên Hệ

**VinUniversity — College of Engineering and Computer Science**  
Hà Nội, Việt Nam
