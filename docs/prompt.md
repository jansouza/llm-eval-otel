Atue como um Engenheiro Sênior de Observabilidade e Plataforma especializado em OpenTelemetry (OTel) e GenAI.

OBJETIVO:
Desenvolver um microserviço/aplicação avaliadora que funcione de forma desacoplada dentro de um pipeline OTel. A aplicação deve consumir spans OTel contendo interações de GenAI (ex: prompts e completions extraídos de spans com gen_ai.*), executar rotinas de avaliação de segurança/qualidade (heurísticas locais como PII/regex e avaliadores estruturados) e emitir métricas e spans/eventos de avaliação 100% aderentes às convenções semânticas do OpenTelemetry (OTel Semantic Conventions for GenAI).

REQUISITOS ARQUITETURAIS:
1. Padrão de Ingestão e Pipeline:
   - A aplicação deve ser capaz de receber dados do OTel Collector (ex: via receiver OTLP/HTTP ou OTLP/gRPC, ou consumindo traces de um tópico Kafka/queue intermediário onde o Collector publica traces).
   - Deve deserializar o payload OTLP (TracesData) preservando o TraceID, SpanID e ParentSpanID originais.

2. Mecanismo de Avaliação (Evaluator Engine):
   - Extrair o conteúdo relevante do span (ex: gen_ai.prompt, gen_ai.completion, input/output do span de chat/LLM).
   - Implementar uma avaliação interna heurística/segurança (exemplo concreto: Detecção de PII com scanner local de regex para CPF, E-mail e Cartão de Crédito).
   - Deixar a interface extensível para plugar outros avaliadores (ex: toxicidade, jailbreak ou LLM-as-a-Judge).

3. Compatibilidade com OTel SemConv GenAI:
   - Para cada avaliação executada, enriquecer a telemetria gerando:
     a) Novo Span filho ou Span Event "gen_ai.evaluation" associado ao TraceID/SpanID original com os atributos padronizados:
        - gen_ai.evaluation.name (ex: "pii_detection")
        - gen_ai.evaluation.type (ex: "heuristic")
        - gen_ai.evaluation.score (ex: 1.0 para detectado, 0.0 para limpo)
        - gen_ai.evaluation.label (ex: "detected" ou "safe")
        - gen_ai.evaluation.explanation (descrição sanitizada sem expor o PII)
        - gen_ai.guardrail.action (ex: "flagged" ou "blocked")
     b) Métricas OTel OTLP exportadas para o backend de métricas:
        - gen_ai.evaluations (Counter: total de avaliações por evaluation.name, evaluation.label, status)
        - gen_ai.evaluation.score (Histogram ou Gauge: distribuição de scores)
   - NUNCA incluir o valor bruto do dado sensível (PII) nos atributos de span ou métricas.

4. Exportação:
   - O microserviço deve utilizar o OTel SDK para exportar os novos traces/spans/eventos e métricas via OTLP (gRPC/HTTP) de volta para o OpenTelemetry Collector ou backend de observabilidade.

ENTREGÁVEIS ESPERADOS:
1. Código completo e modular (recomendo Python com FastAPI + opentelemetry-sdk/opentelemetry-proto, ou Go).
2. Tratamento de payloads OTLP e extração correta de atributos gen_ai.*.
3. Configuração de exemplo do otel-collector-config.yaml demonstrando como encaminhar os spans para esta aplicação avaliadora e receber as métricas de volta.
4. Instruções claras de execução e teste unitário simulando a recepção de um span GenAI e a validação das métricas/spans gerados.