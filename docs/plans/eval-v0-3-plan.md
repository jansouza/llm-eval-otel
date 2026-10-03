# Plano — avaliadores v0.3

30/09/2026 · Jan Souza

> **Estado (01/10/2026).** A versão 0.3.0 entrega o `relevance` e tudo de que ele depende: faixa de execução, cliente do juiz com o adaptador `openai`, orçamento de tokens, mascaramento, telemetria do juiz, descarte de `self_telemetry`, `context_messages`, `tools/benchmark.py`, servidor falso e demonstração (etapas 2 a 6 e 8, e a parte da etapa 1 que não é de retrieval). Falta a calibração com rótulos humanos e modelos reais, então o limiar de `relevance` ainda é o inicial. O `faithfulness` (etapa 7, a rota agrupada e os nomes de retrieval da etapa 1) fica para uma versão futura.

## Contexto

Este plano cobre a segunda onda do Roadmap de avaliadores da [spec](../spec.md): avaliadores em que um LLM julga a interação.

| Avaliador | Pergunta ao juiz | `sample_rate` |
| --- | --- | --- |
| `relevance` | A resposta atende ao que o usuário pediu? | 0.05 |
| `faithfulness` | As afirmações da resposta estão apoiadas nos documentos recuperados? | 0.05 |

Resultado esperado:

- Os dois avaliadores, calibrados contra rótulos humanos, rodando numa fração dos traces com custo previsível e limitado.
- Nenhum dado sensível detectável sai do serviço para o provedor do juiz, e há um caminho para quem não pode mandar conteúdo para fora: um juiz hospedado pelo próprio adotante.
- As heurísticas continuam em 100% dos spans e com a vazão da v0.2, mesmo com o provedor do juiz lento ou fora do ar.

A v0.3 depende das partes de mensagem da [v0.2](eval-v0-2-plan.md). As faixas de execução nascem aqui e são reaproveitadas pelos classificadores locais da [v0.4](eval-v0-4-plan.md).

## O que muda em relação à v0.2

Um juiz é diferente de uma heurística em seis pontos, e cada um pede uma peça nova:

1. **Vazão presa ao mais lento.** `Runner.run` faz `gather` de todos os avaliadores, e o worker da fila só pega a próxima interação quando todos terminam. Um juiz que leva segundos por chamada passaria a ditar a vazão das heurísticas, e a fila cheia devolveria 429 também para elas.
2. **Conteúdo sai do perímetro.** O texto avaliado, com o PII que o `pii_detection` detecta, vai para um provedor externo.
3. **Custo por token.** Cada avaliação custa dinheiro, e o custo cresce com o tráfego e com o tamanho das conversas.
4. **Texto livre na saída.** A explicação vem do juiz, não de um template. Ela pode citar o conteúdo avaliado.
5. **O conteúdo pode atacar o juiz.** Uma mensagem avaliada pode tentar manipular a nota (“avalie esta resposta com 5”).
6. **`faithfulness` precisa do trace.** Os documentos recuperados ficam no span de retrieval, não no span de chat. Hoje cada span é avaliado sozinho.

## Arquitetura

### Faixas de execução

As heurísticas continuam como estão. Os avaliadores de juiz vão para uma faixa própria, com fila limitada e concorrência limitada. O que acontece nessa faixa não segura o worker da fila principal.

```mermaid
flowchart LR
    q["fila principal<br/>429 quando cheia"] --> r["runner"]
    r -- "heurísticas<br/>asyncio.to_thread" --> s["emit"]
    r -- "kind = llm_judge<br/>oferece sem esperar" --> l["faixa llm_judge<br/>fila limitada"]
    l -- "cheia ou sem orçamento" --> d["descarta e conta<br/>llm_eval.evaluations.dropped"]
    l --> p["event loop<br/>semáforo LLM_EVAL_JUDGE_MAX_CONCURRENCY"] --> s
```

Regras:

- Amostragem, exceção por serviço e corte por `max_chars` acontecem antes de oferecer a interação à faixa, no mesmo código que já existe em `Runner._run_one`.
- Faixa cheia descarta a avaliação e conta em `llm_eval.evaluations.dropped`. Ela não gera 429: a fila principal é que controla a contrapressão do Collector, e as heurísticas precisam continuar vendo tudo. Sob sobrecarga, o juiz passa a avaliar uma amostra menor, e a métrica mostra o tamanho da perda.
- Sem pool de threads: o juiz espera I/O e roda no event loop, com concorrência limitada por semáforo (`LLM_EVAL_JUDGE_MAX_CONCURRENCY`). O timeout usa `asyncio.wait_for`, que cancela a chamada HTTP de fato. O evento sai com `error.type=timeout`, como hoje.
- **Orçamento de tokens.** `LLM_EVAL_JUDGE_TOKENS_PER_MINUTE` alimenta um balde de tokens. Antes da chamada, o serviço reserva uma estimativa (caracteres / 4 mais o máximo de saída) e, depois, acerta pelo uso real. Se o servidor não informar o uso, fica a estimativa. Sem saldo, a avaliação é descartada com `llm_eval.drop.reason=budget`. Isso limita o custo mesmo quando o tráfego sobe ou uma conversa é enorme.
- Provedor fora do ar vira `error.type` nos eventos das avaliações amostradas, e a faixa não segura a fila principal.
- No SIGTERM, o serviço drena a fila principal e depois as faixas, dentro dos mesmos 30 s. O que sobra conta em `llm_eval.evaluations.dropped` com o motivo `shutdown`.
- `/readyz` continua olhando só a fila principal.
- A faixa é genérica por `kind`. A v0.4 acrescenta a faixa `model`, com pool de threads no lugar do semáforo, porque inferência local é trabalho de CPU.

Novos sinais de auto-observabilidade:

| Sinal | Tipo | Atributos |
| --- | --- | --- |
| `llm_eval.evaluations.dropped` | Counter | `gen_ai.evaluation.name`, `llm_eval.drop.reason` (`lane_full`, `budget`, `shutdown`) |
| `llm_eval.lane.size` | UpDownCounter | `llm_eval.lane` (`llm_judge`) |

### Cliente do juiz

Um protocolo `JudgeClient` em `judge/client.py`, implementado pelo adaptador `openai` e por um cliente falso nos testes. O avaliador não conhece o provedor nem o SDK.

```python
@dataclass(frozen=True)
class JudgeResponse:
    output: Mapping[str, Any]        # já validado contra o schema
    model: str                       # modelo que respondeu, não o pedido
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    finish_reason: str

class JudgeClient(Protocol):
    async def judge(self, system: str, content: str, schema: Mapping[str, Any]) -> JudgeResponse: ...
```

- **Um adaptador, `openai`.** SDK oficial (`openai`, cliente `AsyncOpenAI`) sobre a API Chat Completions, que é a que os servidores compatíveis implementam. `LLM_EVAL_JUDGE_BASE_URL` vazio usa a API da OpenAI; preenchido aponta para qualquer servidor compatível: vLLM, Ollama ou um gateway como o LiteLLM na frente de outros provedores. O mesmo código cobre o juiz na nuvem e o juiz hospedado pelo adotante, quando o conteúdo não pode sair da rede.
- **Saída estruturada.** `response_format` com JSON schema em modo `strict`, o que dispensa parsear texto livre. Nem todo servidor compatível aceita: `LLM_EVAL_JUDGE_RESPONSE_FORMAT` cai para `json_object` ou `none`, e nesses modos o schema também vai descrito no prompt. Em todos os modos o serviço valida a resposta contra o schema antes de usar.
- **Término.** `message.refusal` preenchido ou `finish_reason` `content_filter` vira `error.type=judge_refusal`; `length` vira `judge_truncated`.
- **Cache.** O prompt do juiz é fixo e vai primeiro, na mensagem de sistema, para formar um prefixo estável. A OpenAI aplica cache de prefixo automaticamente acima de um tamanho mínimo e informa `usage.prompt_tokens_details.cached_tokens`; a calibração confere se o cache pegou. Servidor que não informa o campo conta 0.
- **Parâmetros opcionais.** Servidores e modelos divergem nos parâmetros que aceitam: modelos de raciocínio recusam `temperature`, e vários servidores recusam `reasoning_effort`. Os dois só vão na chamada quando configurados (`LLM_EVAL_JUDGE_TEMPERATURE`, `LLM_EVAL_JUDGE_REASONING_EFFORT`).
- **Retentativas.** O SDK já repete 429 e 5xx; aqui fica com `max_retries=1` e timeout dentro do `timeout_s` do avaliador.

**Modelo.** `LLM_EVAL_JUDGE_MODEL` é obrigatório quando um avaliador de juiz está habilitado. Sem padrão silencioso, porque o modelo define custo e qualidade. A calibração compara no mesmo conjunto pelo menos um modelo grande e um pequeno da API da OpenAI e um modelo aberto servido localmente pelo vLLM ou pelo Ollama, e registra concordância e custo por avaliação de cada um. A troca de modelo é decisão de quem opera, com esses números na mão. O ID do modelo é fixo, com versão ou data quando o provedor oferece, sem alias que mude por baixo.

**API de lotes.** A Batch API da OpenAI custa metade, mas devolve resultados de forma assíncrona e exigiria guardar o estado dos lotes pendentes. Isso contraria o requisito de serviço sem estado da spec. Fica fora da v0.3 e registrado como opção para um worker separado.

### Exceção por serviço

Para as heurísticas, o runner roda o avaliador mesmo num serviço liberado, para mostrar quanto dado sensível ele envia. Para um juiz, isso significaria pagar e mandar conteúdo para fora sem necessidade. Por isso o runner muda para `kind = llm_judge`: serviço liberado não chama o juiz, e o evento sai com `label` `exempt` e explicação `exempt service; not evaluated`.

### Privacidade do conteúdo enviado

- **Mascaramento antes de enviar.** `find_pii` e `find_secrets` já devolvem posições. O texto vai ao juiz com cada ocorrência trocada pelo tipo (`[CPF]`, `[EMAIL]`, `[SECRET]`), o que preserva a estrutura para o julgamento. Ligado por padrão (`LLM_EVAL_JUDGE_REDACT=true`). Nomes e endereços não são mascarados até o `pii_ner` da v0.4.
- **Juiz local.** O mesmo adaptador apontado para um servidor na rede do adotante (`LLM_EVAL_JUDGE_BASE_URL`) cobre quem não pode mandar conteúdo para fora.
- **README.** Uma seção “o que sai para o provedor do juiz”, com o que é mascarado e o que não é.

### Explicação do juiz

- O schema pede `reason` com no máximo 300 caracteres, e o prompt instrui a não citar o conteúdo. O serviço corta em 300 caracteres de qualquer forma e passa pelo sanitizador, que já existe para este caso.
- O sanitizador pega PII e credenciais por regex, não nomes. Um nome citado na justificativa passa por ele. A v0.4 acrescenta a opção de passar a justificativa também pelo `pii_ner`.
- `LLM_EVAL_JUDGE_EXPLANATION=false` troca a explicação por um template (`score=4/5`), para quem não quer texto livre na telemetria.

### Ataque ao juiz

- O conteúdo vai numa seção delimitada (`<conversation>...</conversation>`) e o prompt do juiz diz que tudo ali é dado a avaliar, nunca instrução.
- A saída estruturada limita a resposta ao schema. Nota fora da faixa ou campo ausente vira `error.type=judge_invalid_output`.
- O risco que sobra, o conteúdo enviesar a nota dentro da faixa, fica documentado. A partir da v0.4, o painel pode cruzar com `prompt_injection` no mesmo span.

### Telemetria do juiz e loop

- Cada chamada ao juiz vira um span `chat {modelo}`, filho do span `evaluate {nome}`, com `gen_ai.operation.name`, `gen_ai.provider.name` (`openai`, a API usada), `server.address`, `server.port`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` e `gen_ai.response.finish_reasons`. `server.address` é o que distingue a OpenAI de um servidor local. Nunca com conteúdo: nada de `gen_ai.input.messages` nem `gen_ai.output.messages`, porque isso copiaria o conteúdo do usuário para o backend sob o nome do avaliador. Os spans são criados à mão; bibliotecas de instrumentação automática do SDK ficam de fora, porque podem gravar conteúdo se uma variável de ambiente estiver ligada.
- Métricas da semconv para o juiz: `gen_ai.client.token.usage` (com `gen_ai.token.type`) e `gen_ai.client.operation.duration`. Os nomes entram em `semconv.py`. O serviço não calcula custo em dinheiro, porque preço muda; o painel multiplica tokens pelo preço vigente.
- **Loop.** A topologia da spec já impede o loop: o serviço exporta para o receptor `otlp/eval`, que não manda nada ao avaliador. Como defesa extra, para o caso de alguém apontar `OTEL_EXPORTER_OTLP_ENDPOINT` para o receptor das aplicações, o extrator descarta spans cujo `service.name` do resource é o do próprio serviço e conta em `llm_eval.spans.skipped` com o motivo novo `self_telemetry`.

Configuração nova:

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `LLM_EVAL_JUDGE_MODEL` | vazio, obrigatório com juiz habilitado | ID do modelo |
| `LLM_EVAL_JUDGE_BASE_URL` | vazio (API da OpenAI) | endpoint compatível com OpenAI, como vLLM, Ollama ou LiteLLM |
| `LLM_EVAL_JUDGE_RESPONSE_FORMAT` | `json_schema` | `json_schema`, `json_object` ou `none`, conforme o que o servidor aceita |
| `LLM_EVAL_JUDGE_TEMPERATURE` | vazio (não enviado) | `temperature` da chamada |
| `LLM_EVAL_JUDGE_REASONING_EFFORT` | vazio (não enviado) | `reasoning_effort`, para modelos de raciocínio |
| `LLM_EVAL_JUDGE_MAX_CONCURRENCY` | `8` | chamadas simultâneas ao juiz |
| `LLM_EVAL_JUDGE_QUEUE_MAX` | `1000` | avaliações na faixa antes de descartar |
| `LLM_EVAL_JUDGE_TOKENS_PER_MINUTE` | vazio (sem limite) | orçamento de tokens |
| `LLM_EVAL_JUDGE_REDACT` | `true` | mascara PII e credenciais antes de enviar |
| `LLM_EVAL_JUDGE_EXPLANATION` | `true` | emite a justificativa do juiz como explicação |

A credencial segue a variável padrão do SDK, `OPENAI_API_KEY`, montada como segredo. Servidores locais sem autenticação aceitam qualquer valor, mas o SDK exige que a variável exista.

## Avaliadores

### `relevance`

- **Entrada.** As mensagens de usuário novas no turno e a resposta. Perguntas de continuação (“e em inglês?”) não fazem sentido sem o turno anterior, então `GenAIInteraction` ganha um campo aditivo, `context_messages: list[Message] = []`, com até as 4 mensagens anteriores ao turno, limitadas em caracteres. O extrator preenche o campo; as heurísticas o ignoram.
- **`applies_to`.** Há mensagem de usuário na entrada e parte `text` na saída. Um passo de agente que só chama ferramenta não é resposta ao usuário e fica de fora; a distinção usa as partes de mensagem da v0.2.
- **Escala.** O juiz dá nota de 1 a 5 com justificativa. O score é `(nota - 1) / 4`, e `pass` vai de 3 para cima (score 0,5 ou mais).
- **Parâmetros.** `sample_rate` 0.05, `max_chars` 16 000 (com `llm_eval.content.truncated`), `timeout_s` 30.
- **Atributos.** `llm_eval.judge.model` e `llm_eval.judge.raw_score`.

### `faithfulness`

- **Entrada.** A resposta e os documentos dos spans de retrieval do mesmo trace (`gen_ai.retrieval.documents`, segundo o roadmap; o nome e a estrutura são confirmados na primeira etapa, no commit de referência da semconv).
- **Escala.** O juiz lista as afirmações da resposta e marca as apoiadas pelos documentos. O score é `apoiadas / total`, e `pass` vai de 0,8 para cima (limiar inicial, a calibrar). Resposta sem afirmação verificável dá `pass` com explicação `no claims`.
- **Parâmetros.** `sample_rate` 0.05, `max_chars` 32 000, `timeout_s` 60. No corte, a resposta fica inteira e os documentos são cortados. Documento cortado pode fazer uma afirmação verdadeira parecer sem apoio, e a marca `llm_eval.content.truncated` permite filtrar esses casos no painel.
- **Fora de escopo.** Aplicações de RAG que colocam os documentos dentro do prompt, sem span de retrieval.

**Como juntar o span de chat com o de retrieval.** Há duas opções:

| | Buffer no serviço | `groupbytrace` no Collector |
| --- | --- | --- |
| Estado | o serviço guarda spans por TraceID durante uma janela | o Collector guarda; o serviço continua sem estado |
| Réplicas | exige `loadbalancing` por TraceID | exige o mesmo, se houver mais de um Collector |
| Código novo | buffer, janela, limite de memória, drenagem | uma rota nova e a montagem da interação |

Recomendação: `groupbytrace`, porque mantém o requisito de serviço sem estado da spec. O Collector ganha um pipeline separado, só para inferência e retrieval, que espera o trace se completar e manda o grupo para uma rota própria do serviço. O pipeline `traces/genai` atual não muda, e as heurísticas não esperam a janela.

```yaml
processors:
  filter/genai_rag:              # mantém inferência e retrieval
    error_mode: ignore
    trace_conditions:
      - >-
        not IsMatch(span.attributes["gen_ai.operation.name"],
        "^(chat|text_completion|generate_content|retrieval)$")   # nome de retrieval a confirmar
  groupbytrace/rag:
    wait_duration: 10s
    num_traces: 10000

exporters:
  otlp_http/evaluator_grouped:
    traces_endpoint: http://llm-eval-otel:4318/v1/traces/grouped
    compression: gzip
    sending_queue: { enabled: true }
    retry_on_failure: { enabled: true }

service:
  pipelines:
    traces/genai_grouped:
      receivers: [otlp]
      processors: [filter/genai_rag, groupbytrace/rag, batch]
      exporters: [otlp_http/evaluator_grouped]
```

No serviço:

- Rota `POST /v1/traces/grouped`. O extrator lê também os spans de retrieval e anexa à interação de chat os documentos dos spans de retrieval do mesmo trace que terminaram antes de o chat começar, num campo aditivo `retrieved_documents: tuple[str, ...] = ()`.
- Um avaliador declara em qual rota roda com o atributo opcional `scope` (`"span"` por padrão, `"trace"` para `faithfulness`), lido com `getattr` para não quebrar o protocolo. Cada rota roda só os avaliadores do seu escopo.
- A deduplicação separa as rotas na chave, porque o mesmo span de chat chega pelas duas.
- Span de retrieval que chega depois da janela deixa o chat sem documentos. O avaliador não se aplica, e o caso conta em `llm_eval.spans.skipped` com o motivo `no_retrieval_context`.

## Calibração

- **Conjunto.** Cerca de 200 interações por avaliador, em português e inglês, rotuladas por pessoas: `pass`/`fail` e nota. Para `faithfulness`, com documentos e respostas que misturam afirmações apoiadas e inventadas. Conteúdo sintético, sem dado real de usuário.
- **Ferramenta.** `tools/benchmark.py` roda um avaliador sobre um JSONL rotulado e relata concordância com o rótulo humano em `pass`/`fail`, variação da nota ao repetir a mesma entrada, latência p50 e p99, tokens e custo estimado por avaliação (com o preço por milhão de tokens passado na linha de comando). A v0.4 reaproveita a ferramenta para validar os classificadores locais.
- **Critério de aprovação (proposta, a fechar com os primeiros números).** Concordância de 80% ou mais com o rótulo humano em `pass`/`fail`, e variação pequena da nota ao repetir a mesma entrada.
- **Saída.** Modelo escolhido, limiar, concordância e custo por mil avaliações registrados neste documento e no README.

## Testes

- **Unitários.** Um `JudgeClient` falso e determinístico, para cobrir mascaramento, orçamento, descarte, recusa, saída inválida, timeout, exceção por serviço e o conteúdo dos spans do juiz. O motor é testado com um avaliador falso lento: faixa cheia, timeout, drenagem no desligamento e vazão das heurísticas.
- **Adaptador.** O adaptador `openai` roda contra um servidor falso compatível com OpenAI em `tools/`, que responde de forma determinística e pode recusar `json_schema`, para cobrir os três modos de `LLM_EVAL_JUDGE_RESPONSE_FORMAT`. Como o adaptador é o mesmo para a OpenAI e para servidores locais, o teste exercita o código de produção.
- **Ponta a ponta.** O mesmo servidor falso sobe no compose. O CI não precisa de chave de API. O README mostra como trocar pela OpenAI ou por um servidor local.
- **Carga.** Com o juiz lento (5 s por chamada) e fora do ar, as heurísticas mantêm a vazão da v0.2.
- **Vazamento.** O teste de vazamento passa a olhar também o que foi enviado ao juiz falso: nenhum CPF, cartão ou credencial dos casos sintéticos chega a ele com `LLM_EVAL_JUDGE_REDACT=true`.

## Etapas

1. **Nomes e modelo de dados.** Confirmar no commit de referência os nomes do span de retrieval e de `gen_ai.retrieval.documents`; `context_messages` e `retrieved_documents` em `GenAIInteraction`; motivo `self_telemetry` no extrator.
   - Pronto quando: os nomes estão em `semconv.py` e no teste de snapshot, e um span do próprio serviço enviado ao receptor é descartado com esse motivo.
2. **Faixas de execução.** Faixa por `kind` com fila limitada, `llm_eval.evaluations.dropped`, `llm_eval.lane.size` e drenagem no desligamento.
   - Pronto quando: com um avaliador falso de 2 s, as heurísticas mantêm pelo menos 95% da vazão sem ele; a faixa cheia conta descarte sem gerar 429; e o timeout gera `error.type=timeout` sem afetar os outros avaliadores.
3. **Cliente do juiz.** `JudgeClient`, adaptador `openai`, cliente falso, servidor falso compatível, spans e métricas do juiz sem conteúdo.
   - Pronto quando: o adaptador passa nos testes contra o servidor falso nos três modos de `LLM_EVAL_JUDGE_RESPONSE_FORMAT`, recusa e saída inválida viram `error.type`, e o span do juiz não tem atributo de conteúdo.
4. **Faixa do juiz, orçamento e exceção.** Faixa `llm_judge`, semáforo, balde de tokens, descarte por `budget`, serviço liberado sem chamada.
   - Pronto quando: com orçamento esgotado as avaliações são descartadas e contadas, e serviço liberado não gera chamada ao juiz.
5. **Mascaramento e explicação.** Mascaramento antes do envio, corte e sanitização da justificativa, `LLM_EVAL_JUDGE_EXPLANATION`.
   - Pronto quando: o juiz falso nunca recebe os valores dos casos sintéticos, e uma justificativa que cita um CPF sai como `[REDACTED]`.
6. **`relevance`.** Avaliador, prompt, schema, `tools/benchmark.py` e calibração.
   - Pronto quando: o avaliador passa no critério de aprovação da calibração com o modelo escolhido, e os números estão registrados.
7. **`faithfulness`.** Rota agrupada, montagem da interação com documentos, escopo `trace`, pipeline no `otel-collector-config.yaml`, avaliador e calibração.
   - Pronto quando: um trace sintético com retrieval e chat gera o evento no span de chat, um chat sem retrieval conta `no_retrieval_context`, e o avaliador passa na calibração.
8. **Demonstração e documentação.** Servidor falso no compose, casos no gerador (resposta relevante e fora do assunto; resposta fiel e inventada), teste de carga, spec atualizada (evento, configuração, Collector, auto-observabilidade, roadmap) e README com custo por mil avaliações e a seção sobre o que sai para o provedor.
   - Pronto quando: o teste ponta a ponta passa no CI sem chave de API e as metas estão registradas no README.

## Critérios de aceite

- [ ] `relevance` e `faithfulness` rodam em cerca de 5% dos traces, sempre nos mesmos TraceIDs, enquanto as heurísticas rodam em todos.
- [ ] Com o juiz lento ou fora do ar, as heurísticas mantêm a vazão da v0.2 e nenhum 429 é causado pelo juiz.
- [ ] Faixa cheia e orçamento esgotado descartam e contam em `llm_eval.evaluations.dropped`, com os motivos `lane_full` e `budget`.
- [ ] O mesmo adaptador funciona com a API da OpenAI e com um servidor local compatível, verificado na calibração.
- [ ] Nenhum valor detectado pelo `pii_detection` ou pelo `secret_detection` chega ao juiz com o mascaramento ligado.
- [ ] Serviço listado em `LLM_EVAL_EXCEPTIONS` para um juiz não gera chamada e sai como `exempt`.
- [ ] A justificativa do juiz passa pelo sanitizador e tem no máximo 300 caracteres.
- [ ] O span da chamada ao juiz tem tokens, modelo, endpoint e motivo de término, e nenhum conteúdo.
- [ ] Recusa do juiz, saída fora do schema e timeout geram evento com `error.type` e severidade `ERROR`.
- [ ] Spans do próprio serviço que voltam ao receptor são descartados com `self_telemetry`.
- [ ] Os dois avaliadores passaram na calibração, com modelo, limiar, concordância e custo registrados.

## Riscos

| Risco | Efeito | Mitigação |
| --- | --- | --- |
| Custo acima do previsto | Conta alta no provedor | `sample_rate` baixo, orçamento de tokens, `max_chars`, cache do prompt do juiz, custo por mil avaliações medido na calibração |
| Conteúdo sensível enviado ao provedor | Exposição fora do perímetro | Mascaramento por padrão; juiz local por `LLM_EVAL_JUDGE_BASE_URL`; seção no README |
| Servidor compatível diverge da API da OpenAI | Saída estruturada recusada, uso ausente, parâmetro rejeitado | `LLM_EVAL_JUDGE_RESPONSE_FORMAT`; validação local do schema em todos os modos; parâmetros opcionais só quando configurados; uso ausente fica na estimativa do orçamento |
| Descarte silencioso do juiz sob carga | Avaliações faltando no painel | `llm_eval.evaluations.dropped` com alerta sugerido no README |
| Nota manipulada pelo conteúdo avaliado | Falso `pass` | Conteúdo delimitado, saída estruturada; cruzamento com `prompt_injection` a partir da v0.4 |
| Provedor muda o comportamento do modelo | Notas mudam sem mudança no serviço | ID de modelo fixo, sem alias; `gen_ai.response.model` e `llm_eval.judge.model` no evento; recalibrar a cada troca |
| Justificativa com nome ou dado não detectado por regex | Dado pessoal na telemetria | Corte em 300 caracteres; `LLM_EVAL_JUDGE_EXPLANATION=false`; sanitização por `pii_ner` na v0.4 |
| `groupbytrace` segura muitos traces | Memória alta no Collector | Filtro só de inferência e retrieval antes do agrupamento; `num_traces` limitado |
| Nomes de retrieval mudam na semconv (status Development) | `faithfulness` deixa de encontrar documentos | Nomes em `semconv.py`, teste de snapshot, motivo `no_retrieval_context` visível nas métricas |

## Decisões em aberto

- **Modelo padrão do juiz.** Sai da calibração, com concordância e custo dos modelos comparados.
- **Provedores com API própria (Anthropic, Gemini, Bedrock).** Sem adaptador nativo. Quem precisar usa o endpoint compatível com OpenAI que o provedor oferece ou um gateway como o LiteLLM. Um adaptador nativo só entra se a compatibilidade não bastar para saída estruturada ou cache.
- **API de lotes.** Metade do custo, mas exige estado. Só vale com um worker separado, fora deste serviço.
- **RAG com documentos no prompt.** Um `faithfulness` que use as mensagens de sistema ou de entrada como fonte cobriria esses casos, mas precisaria separar documentos de instruções. Fica para depois de ver como as aplicações reais gravam o contexto.
