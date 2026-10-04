# Phoenix AI Real-World E2E Evaluation v3

**Result:** 9/14 passed (64.3%)

## RAG-TXT-01: PASS
- Latency: 17.605s
- Intent: rag
- Agent: rag
## RAG-TXT-02: PASS
- Latency: 4.62s
- Intent: rag
- Agent: rag
## RAG-DOCX-01: FAIL
- Latency: 3.69s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: qwen3:4b-instruct
## RAG-PDF-01: FAIL
- Latency: 6.37s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: nomic-embed-text
## RAG-PPTX-01: FAIL
- Latency: 7.947s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: approval; unexpected RAG status: 'unverified'
## RAG-XLSX-01: FAIL
- Latency: 7.043s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: 768; unexpected RAG status: 'unverified'
## RAG-XLSX-02: FAIL
- Latency: 8.144s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: engine a; missing expected term: 14400; unexpected RAG status: 'unverified'
## RAG-UNKNOWN-01: PASS
- Latency: 4.098s
- Intent: rag
- Agent: rag
## PLAN-01: PASS
- Latency: 117.079s
- Intent: planning
- Agent: planning
## APP-01: PASS
- Latency: 0.454s
- Intent: application
- Agent: application
## SEC-01: PASS
- Latency: 0.029s
- Intent: application
- Agent: application
## DOC-ROUTING-01: PASS
- Latency: 3.694s
- Intent: rag
- Agent: rag
## MEM-STORE: PASS
- Latency: 0.023s
- Intent: memory
- Agent: memory
## MEM-RECALL: PASS
- Latency: 0.046s
- Intent: memory
- Agent: memory