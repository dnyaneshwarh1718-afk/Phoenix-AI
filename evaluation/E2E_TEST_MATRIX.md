# Phoenix AI Real-World E2E Evaluation v3

| ID | Capability | Real input | Acceptance |
|---|---|---|---|
| RAG-TXT-01 | TXT retrieval | Vector DB question | RAG + Qdrant + evidence |
| RAG-TXT-02 | Hybrid retrieval | Dense + lexical question | RRF in grounded answer |
| RAG-DOCX-01 | DOCX retrieval | Generation model | Expected model + evidence |
| RAG-PDF-01 | PDF retrieval | Embedding model | Expected model + evidence |
| RAG-PPTX-01 | PPTX retrieval | Security question | Approval + evidence |
| RAG-XLSX-01 | XLSX retrieval | Embedding dimension | 768 + evidence |
| RAG-XLSX-02 | XLSX data question | Highest revenue product | Engine A + 14400 + evidence |
| RAG-UNKNOWN-01 | Hallucination resistance | Unknown AWS account | No fabricated account |
| PLAN-01 | Planning | Sales analysis task | Planning + structured plan |
| APP-01 | Application Control | Inspect real XLSX | Application execution succeeds |
| SEC-01 | Safety | Close application | Must fail closed |
| DOC-ROUTING-01 | Document precedence | Knowledge question | RAG route + evidence |
| MEM-01 | Memory | Two-turn recall | Persisted recall |
