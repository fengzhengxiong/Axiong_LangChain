一、系统架构设计
RAG Architecture
（示意图，实际为本地化实现）

1:分层架构：
- **应用层**：main.py（交互入口）
- **业务逻辑层**：rag_system.py（流程控制）
- **服务层**：llm_model.py/vector_store.py/document_processor.py
- **基础设施层**：config.py（配置中心）

2:数据流：
- **流程图如下:**
- **A[用户输入] --> B[文档加载]**
- **A[用户输入] --> B[文档加载]**
- **B --> C[文本分割]**
- **C --> D[向量化存储]**
- **A --> E[向量检索]**
- **D --> E**
- **E --> F[Prompt构建]**
- **F --> G[LLM生成]**
- **G --> H[输出答案]**

---
- **开发人员**：冯政雄
- **邮箱**：1220597071@qq.com
- **简介**：langchain框架学习记录

---
- **向量数据库**：FAISS
- **向量模型**：nomic-embed-text:latest
- **llm模型**：deepseek-r1:8b

---
- **Q: 为什么要选用FAISS？**：
- **A: Weaviate需要docker镜像，然后用python 去连接docker**
- **A: Milvus电脑带不动**
- **A: LanceDB，我想区分开anythingllm**
- **A: FAISS在本文中使用的是cpu版，别问我为什么不用gpu，问就是穷**

---
- **Q: 为什么要选用nomic-embed-text:latest？**
- **A: 它比较小，且看论坛说比较牛逼**

---
- **Q: 为什么要选用deepseek-r1:8b？**
- **A: 我喜欢**

---
- **Q: 能运行么？**
- **A: 包的**

