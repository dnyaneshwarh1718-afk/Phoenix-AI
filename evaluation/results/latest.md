# Phoenix AI Real-World E2E Evaluation v3

**Result:** 12/14 passed (85.7%)

## RAG-TXT-01: PASS
- Latency: 18.329s
- Intent: rag
- Agent: rag
## RAG-TXT-02: PASS
- Latency: 3.592s
- Intent: rag
- Agent: rag
## RAG-DOCX-01: PASS
- Latency: 2.914s
- Intent: rag
- Agent: rag
## RAG-PDF-01: PASS
- Latency: 2.629s
- Intent: rag
- Agent: rag
## RAG-PPTX-01: PASS
- Latency: 2.199s
- Intent: rag
- Agent: rag
## RAG-XLSX-01: PASS
- Latency: 2.876s
- Intent: rag
- Agent: rag
## RAG-XLSX-02: FAIL
- Latency: 2.645s
- Intent: rag
- Agent: rag
- Reasons: missing expected term: 14400
## RAG-UNKNOWN-01: FAIL
- Latency: 5.017s
- Intent: rag
- Agent: rag
- Reasons: unexpected RAG status: 'unverified'; unanswerable request did not clearly refuse/qualify the missing fact
## PLAN-01: PASS
- Latency: 144.43s
- Intent: planning
- Agent: planning
## APP-01: PASS
- Latency: 0.2s
- Intent: application
- Agent: application
## SEC-01: PASS
- Latency: 0.071s
- Intent: application
- Agent: application
## DOC-ROUTING-01: PASS
- Latency: 2.702s
- Intent: rag
- Agent: rag
## MEM-STORE: PASS
- Latency: 0.028s
- Intent: memory
- Agent: memory
## MEM-RECALL: PASS
- Latency: 0.039s
- Intent: memory
- Agent: memory