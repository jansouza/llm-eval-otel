# Plano — avaliadores v0.2

30/09/2026 · Jan Souza

## Contexto

Este plano cobre a primeira onda do Roadmap de avaliadores da [spec](../spec.md): heurísticas locais, sem modelo, que rodam em 100% dos spans como `pii_detection` e `secret_detection`.

| Entrega | Avaliador | Tipo |
| --- | --- | --- |
| PII ampliado: CNPJ, telefone e chave PIX | `pii_detection` (existente) | `heuristic` |
| Recusa do modelo | `refusal` (novo) | `heuristic` |
| Vazamento das instruções de sistema | `system_prompt_leak` (novo) | `heuristic` |
| Formato da saída | `output_format` (novo) | `heuristic` |

Resultado esperado:

- O `pii_detection` reconhece CNPJ (numérico e alfanumérico), telefone brasileiro e chave PIX aleatória, sem aumentar os falsos positivos dos casos atuais.
- Três avaliadores novos, registrados por entry point e habilitados por `LLM_EVAL_EVALUATORS`, cada um emitindo evento, span filho e métricas como os da v0.1.
- Nenhum valor avaliado em explicação, atributo, métrica ou log, com a mesma regra de sanitização da v0.1.
- As metas de desempenho da v0.1 continuam valendo: p99 de até 5 ms por avaliador para 10 KB de texto.

## O que muda em relação à spec

O roadmap diz que nenhuma das quatro entregas muda a arquitetura. Isso vale para o runner, a fila e o emissor. O modelo de dados precisa de três campos a mais, porque o extrator descarta hoje informação que três dos avaliadores usam:

- `output_format` precisa de `gen_ai.output.type`, que o extrator não lê.
- `output_format` e `system_prompt_leak` precisam separar as partes de uma mensagem. `Message.text` concatena `text`, `reasoning`, `tool_call` e `tool_call_response`; um bloco de raciocínio antes do JSON quebra a validação, e raciocínio que cita as instruções de sistema não chega ao usuário.
- `refusal` usa `gen_ai.response.finish_reasons`: `content_filter` é recusa do provedor, mesmo sem frase de recusa no texto.

As mudanças são aditivas. Os campos novos têm valor padrão e ficam no fim das dataclasses, então avaliadores de terceiros e as heurísticas atuais continuam funcionando sem alteração.

```python
@dataclass(frozen=True)
class PartSpan:
    type: str                        # text | reasoning | tool_call | tool_call_response
    start: int                       # posição em Message.text
    end: int

@dataclass(frozen=True)
class Message:
    role: str
    text: str
    parts: tuple[PartSpan, ...] = ()  # vazio = texto sem divisão conhecida

    def text_of(self, *types: str) -> str: ...  # junta só as partes pedidas

@dataclass(frozen=True)
class GenAIInteraction:
    ...                              # campos atuais, sem mudança
    output_type: str | None = None   # gen_ai.output.type
    finish_reasons: tuple[str, ...] = ()  # gen_ai.response.finish_reasons
```

As partes guardam posições dentro de `text`, não cópias do texto, para não dobrar a memória de cada interação na fila (até `LLM_EVAL_QUEUE_MAX` interações). No formato OpenLLMetry, `content` vira parte `text` e `tool_calls.{n}.arguments` vira parte `tool_call`. Os nomes novos (`gen_ai.output.type`, `gen_ai.response.finish_reasons`) entram em `semconv.py` e no teste de snapshot de nomes.

## Entregas

### 1. PII ampliado, dentro do `pii_detection`

Os tipos novos seguem o desenho dos atuais: padrão, validação extra e, para números sem formatação, uma palavra-chave por perto.

| Tipo | Padrão | Validação extra |
| --- | --- | --- |
| CNPJ (`cnpj`) | Formatado (`XX.XXX.XXX/XXXX-XX`, com letras maiúsculas ou dígitos nas 12 primeiras posições), ou 14 caracteres seguidos com a palavra “CNPJ” até 30 caracteres antes | dois dígitos verificadores (módulo 11, pesos 5..2 e 9..2, depois 6..2 e 9..2); cada caractere vale `ord(c) - 48`, o que cobre o CNPJ alfanumérico; rejeita 14 caracteres iguais |
| Telefone (`phone`) | `+55` opcional, DDD com ou sem parênteses, celular com 9 dígitos começando em 9 ou fixo com 8 dígitos começando de 2 a 5, com espaço ou hífen opcionais; dígitos sem formatação só com “tel”, “telefone”, “celular”, “whatsapp” ou “fone” até 30 caracteres antes | DDD na lista de DDDs válidos da Anatel |
| Chave PIX aleatória (`pix_key`) | UUID v4, sem diferenciar maiúsculas, com a palavra “pix” até 40 caracteres antes | nenhuma |

Notas:

- O CNPJ alfanumérico vale para inscrições novas desde julho de 2026 (IN RFB 2.229/2024). Os dígitos verificadores continuam numéricos, e o mesmo cálculo serve para os dois formatos.
- Chaves PIX que são CPF, CNPJ, e-mail ou telefone já contam no tipo delas. `pix_key` cobre só a chave aleatória, e só com a palavra “pix” por perto: UUID solto continua sem detecção, como pede o critério de aceite do `secret_detection`.
- Quando dois tipos casam o mesmo trecho, conta um só, nesta ordem: CPF, CNPJ, cartão, telefone. Os tipos com dígito verificador vêm primeiro porque erram menos.
- Os telefones de `test_ignores` em `tests/unit/test_pii.py` passam a ser detectados. Esses casos vão para `test_detects` como `phone`, e números de pedido e protocolo continuam em `test_ignores`.
- `emit/sanitize.py` chama `find_pii`, então o sanitizador passa a cobrir os tipos novos sem mudança. O teste do sanitizador ganha casos com valores legítimos que não podem virar `[REDACTED]`: `gen_ai.response.id` como `chatcmpl-...`, nomes de modelo e as explicações do próprio serviço.

**Mudança de comportamento.** Telefone é comum em chatbots de atendimento, e quem já usa o `pii_detection` vai ver mais `fail`. Liberar o avaliador inteiro por `LLM_EVAL_EXCEPTIONS` também desligaria CPF e cartão. Por isso esta entrega traz `LLM_EVAL_PII_TYPES`, com todos os tipos por padrão, para desligar tipos específicos. O sanitizador ignora essa variável e sempre usa todos os tipos.

### 2. `refusal`

Detecta respostas em que o modelo se recusa a atender o pedido.

- **O que lê.** Só as partes `text` das mensagens de saída, porque raciocínio e chamada de ferramenta não chegam ao usuário. Todas as saídas quando `n > 1`.
- **Como.** Frases de recusa em português, inglês e espanhol, em `evaluators/refusal_phrases.py`, procuradas nos primeiros 300 caracteres de cada mensagem, depois de normalizar caixa e acentos (`unicodedata`, NFKD). Recusas abrem a resposta, e limitar ao começo evita casar “não posso deixar de mencionar” no meio de uma resposta normal. As frases exigem verbo de recusa com objeto (“não posso ajudar com”, “I can't assist with”, “no puedo ayudar con”), não só “não posso”.
- **Recusa do provedor.** `content_filter` em `finish_reasons` conta como recusa, mesmo sem texto.
- **Resultado.** `score` 0.0 e `label` `fail` com recusa; 1.0 e `pass` sem. Aqui `fail` quer dizer que o modelo recusou, não que ele errou: recusar um pedido abusivo é o comportamento certo. O painel lê a taxa de recusa por modelo e serviço, e a v0.3 cruza com `prompt_injection`.
- **Explicação.** `refusal=1 (output), source=phrase, lang=pt`.
- **Atributos.** `llm_eval.refusal.source` (`phrase` ou `finish_reason`) e `llm_eval.refusal.language`.
- **`applies_to`.** Há ao menos uma parte `text` de saída, ou `finish_reasons` não está vazio.
- **Fora de escopo.** Recusa parcial (responde com ressalvas) e idiomas além dos três.

### 3. `system_prompt_leak`

Detecta respostas que reproduzem trechos das instruções de sistema.

- **`applies_to`.** O span tem instruções de sistema com pelo menos 30 palavras e ao menos uma parte `text` ou `tool_call` de saída. Instruções curtas, como “Você é um assistente útil.”, não são segredo e dariam sobreposição trivial.
- **O que lê.** Partes `text` e `tool_call` de saída, porque uma chamada de ferramenta pode levar as instruções para fora. Raciocínio fica de fora.
- **Como.**
  1. Normaliza os dois textos: NFKC, caixa baixa, pontuação removida, espaços colapsados.
  2. Monta o conjunto de 8-gramas de palavras das instruções de sistema.
  3. Percorre os 8-gramas da saída e mede a cobertura (fração dos 8-gramas das instruções que aparecem na saída) e a maior sequência de palavras seguidas em comum.
  4. `fail` quando a sequência tem 20 palavras ou mais, ou a cobertura passa de 0,15. Os dois limiares são iniciais e serão calibrados com os casos de teste.
- **Score.** `1 - cobertura`. O rótulo segue os limiares acima, então uma sequência longa com cobertura baixa pode dar `fail` com score alto. O limiar fica documentado, como a spec pede para avaliadores com escala própria.
- **Explicação.** `coverage=0.42, longest_run=37 words (output)`: só números.
- **Atributos.** `llm_eval.prompt_leak.coverage` (double) e `llm_eval.prompt_leak.longest_run` (int).
- **Custo.** Linear no tamanho dos textos, com hash de n-gramas. Cabe na meta de 5 ms.
- **Limites conhecidos.** Não pega paráfrase nem tradução das instruções. Instruções de sistema que trazem conteúdo que o modelo deve repetir (FAQ, textos padrão de atendimento) geram `fail` legítimo pela regra e indevido na prática; o caminho é `LLM_EVAL_EXCEPTIONS` para esses serviços.
- **Pré-requisito.** A aplicação precisa gravar `gen_ai.system_instructions` ou mensagens com papel `system`. Sem isso o avaliador não se aplica e não emite nada.

### 4. `output_format`

Valida que a resposta é JSON quando o cliente pediu JSON.

- **`applies_to`.** `output_type == "json"` e ao menos uma parte `text` de saída.
- **Como.** `json.loads` nas partes `text` de cada mensagem de saída, uma por uma. Sem remover cercas de Markdown: com `gen_ai.output.type = json` o provedor foi chamado em modo JSON, e uma cerca indica que o modo não foi respeitado. Texto vazio é `fail`.
- **Resultado.** `score` = fração de saídas válidas; `label` `fail` se alguma for inválida.
- **Explicação.** `invalid_json=1 of 2 (output): Expecting ',' delimiter at char 132`. Usa `JSONDecodeError.msg` e `.pos`, nunca `.doc`, que é o texto avaliado. Com `length` em `finish_reasons`, a explicação acrescenta `finish_reason=length`, a causa mais comum de JSON cortado.
- **Atributo.** `llm_eval.output_format.error`: `syntax`, `empty` ou `truncated`.
- **Schema.** No commit de referência da semconv, nenhum atributo traz o schema pedido pelo cliente, então esta entrega valida só a sintaxe. Validar contra schema fica para quando houver de onde ler o schema. A primeira etapa confirma isso no commit de referência.
- **OpenLLMetry.** Verificar se o formato grava algum atributo equivalente a `gen_ai.output.type`. Se não gravar, o avaliador não se aplica a esses spans, e o README diz isso.

## Habilitação

`LLM_EVAL_EVALUATORS` continua com `pii_detection,secret_detection` por padrão. Os três avaliadores novos entram por opção, para não mudar o volume de eventos de quem já usa o serviço. O PII ampliado é a exceção: está dentro do `pii_detection` e vem ligado, com `LLM_EVAL_PII_TYPES` para desligar tipos.

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `LLM_EVAL_PII_TYPES` | `cpf,cnpj,email,credit_card,phone,pix_key` | tipos que o `pii_detection` reporta; o sanitizador usa sempre todos |

## Etapas

Cada etapa termina com lint, `mypy --strict` e testes passando, e pode ir para revisão sozinha. A etapa 2 não depende da 1 e pode andar em paralelo.

1. **Modelo de dados e extrator.** `PartSpan`, `Message.parts`, `output_type` e `finish_reasons` em `evaluators/base.py`; leitura em `extract/genai.py` nos dois formatos; nomes em `semconv.py`. Confirmar no commit de referência os nomes lidos e a ausência de atributo de schema.
   - Pronto quando: as fixtures dos dois formatos produzem as mesmas partes para o mesmo diálogo, uma mensagem com raciocínio e texto separa as duas partes, e os testes atuais passam sem alteração.
2. **PII ampliado.** CNPJ, telefone e `pix_key` em `evaluators/pii.py`, a ordem de prioridade entre tipos e `LLM_EVAL_PII_TYPES`.
   - Pronto quando: a tabela de casos passa (CNPJ numérico e alfanumérico com DV certo e errado, 14 dígitos sem a palavra “CNPJ”, telefones fixo e celular em vários formatos, DDD inválido, UUID com e sem “pix”, números de pedido), os telefones antigos de `test_ignores` estão em `test_detects`, e o p99 fica em até 5 ms para 10 KB.
3. **`refusal`.** Avaliador, tabela de frases e entry point.
   - Pronto quando: recusas nos três idiomas dão `fail`, `content_filter` sem texto dá `fail` com `source=finish_reason`, e os quase-acertos (“não posso deixar de”, “I can't wait”, recusa no meio da resposta) dão `pass`.
4. **`system_prompt_leak`.** Avaliador e entry point.
   - Pronto quando: saída que copia um parágrafo das instruções dá `fail`, saída que repete uma frase curta comum dá `pass`, cópia só no raciocínio dá `pass`, e cópia nos argumentos de uma tool call dá `fail`.
5. **`output_format`.** Avaliador e entry point.
   - Pronto quando: JSON válido dá `pass`; JSON cortado com `finish_reason=length` dá `fail` com `error=truncated`; JSON em cerca de Markdown dá `fail`; raciocínio antes do JSON não atrapalha; span sem `gen_ai.output.type` não gera evento.
6. **Demonstração, documentação e versão.** Casos novos no `tools/span_generator.py` (CNPJ, telefone, PIX, recusa em português, vazamento de instruções, JSON válido e inválido); `LLM_EVAL_EVALUATORS` com os cinco avaliadores no `docker-compose.yaml`; teste ponta a ponta; spec atualizada nas tabelas do evento e de configuração e no roadmap; README com os avaliadores novos e o pré-requisito de gravar as instruções de sistema; versão do pacote nova, que vira o `service.version` e identifica as regras.
   - Pronto quando: o teste ponta a ponta encontra no arquivo do exportador `file` um evento de cada avaliador novo, e nenhum valor dos casos sintéticos aparece na saída serializada.

## Critérios de aceite

- [ ] CNPJ numérico e alfanumérico com dígitos verificadores válidos, telefone com DDD válido e chave PIX aleatória com a palavra “pix” por perto geram `fail` no `pii_detection`, com o tipo em `llm_eval.pii.types`.
- [ ] Não são detectados: CNPJ com dígito verificador errado, 14 dígitos sem a palavra “CNPJ” por perto, telefone com DDD inexistente e UUID sem a palavra “pix”.
- [ ] Com `LLM_EVAL_PII_TYPES=cpf,cnpj,email,credit_card`, telefone dá `pass`, e o sanitizador continua redigindo telefone em atributos.
- [ ] Recusa em português, inglês ou espanhol, ou `finish_reasons` com `content_filter`, gera `fail` no `refusal`; resposta normal que contém “não posso” no meio gera `pass`.
- [ ] Saída que reproduz um trecho longo das instruções de sistema gera `fail` no `system_prompt_leak`; span sem instruções de sistema não gera evento desse avaliador.
- [ ] Saída inválida com `gen_ai.output.type = json` gera `fail` no `output_format`, com a posição do erro e sem trecho do texto na explicação.
- [ ] Nenhum valor avaliado aparece em eventos, spans, métricas ou logs; o teste de vazamento da v0.1 cobre os três avaliadores novos.
- [ ] `pii_detection` e cada avaliador novo têm p99 de até 5 ms para 10 KB de texto no teste de carga.
- [ ] Os testes e critérios de aceite da v0.1 continuam passando.

## Riscos

| Risco | Efeito | Mitigação |
| --- | --- | --- |
| Telefone gera muitos `fail` em chatbots de atendimento | Alerta indevido e pressão para liberar o `pii_detection` inteiro | `LLM_EVAL_PII_TYPES`; nota no changelog e no README |
| Número de 11 a 14 dígitos casa telefone ou CNPJ por acaso | Falso positivo | Dígitos sem formatação só com palavra-chave por perto; DDD validado; CNPJ com dígito verificador |
| Frases de recusa variam por modelo e versão | Recusa não detectada | Tabela de frases versionada, casos de teste por provedor, revisão quando um modelo novo entrar nos painéis |
| Instruções de sistema com conteúdo que deve ser repetido | `fail` indevido no `system_prompt_leak` | Limiares calibrados; `LLM_EVAL_EXCEPTIONS` por serviço |
| Aplicações não gravam instruções de sistema nem `gen_ai.output.type` | Dois avaliadores sem ter o que avaliar | Pré-requisito no README; os avaliadores não emitem evento quando não se aplicam, sem ruído |
| Semconv muda os nomes lidos | Campos novos deixam de ser preenchidos | Nomes em `semconv.py` e no teste de snapshot, como na v0.1 |

## Decisões em aberto

- **Métrica de “não se aplica”.** `system_prompt_leak` e `output_format` não emitem nada quando o span não tem os dados. Um contador dessas ocorrências mostraria quantas aplicações não gravam o que eles precisam. Recomendação: esperar o primeiro uso real e decidir com base nele.
- **Validação por schema no `output_format`.** Depende de a semconv ou o OpenLLMetry passarem a gravar o schema pedido. A alternativa é um schema por serviço em configuração, que exige a dependência `jsonschema`. Recomendação: não fazer agora.
