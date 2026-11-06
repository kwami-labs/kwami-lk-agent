# Persistent memory

Each Kwami can keep independent, long-lived memory on
[Zep Cloud](https://www.getzep.com/). Memory is an enhancement, not a
requirement: if `ZEP_API_KEY` is unset, or Zep is slow, the agent still
greets and talks.

Enable it by setting `ZEP_API_KEY`. `Settings.memory_enabled` is true exactly
when that key is present.

## What is stored

| Store | Contents |
| --- | --- |
| Thread | User and assistant turns, batched as one `add_messages` call |
| Knowledge graph | Facts extracted from user turns, with temporal validity |
| Custom ontology | Person, Project, Product, Goal, Procedure plus constrained edges |

Assistant text is sent with the user turn so Zep has context for extraction,
but `ignore_roles=["assistant"]` keeps the assistant from becoming a graph
entity.

```mermaid
flowchart LR
    User["User utterance"] --> Hook["on_user_turn_completed"]
    Hook --> Buf["buffer_user_message"]
    Asst["Assistant turn"] --> Ev["conversation_item_added"]
    Ev --> Ex["add_exchange"]
    Buf --> Ex
    Ex --> Zep["Zep thread + graph"]
    Zep --> Ctx["get_context / get_user_name"]
    Ctx --> Prompt["system prompt addition, sized by the retrieval preset"]
```

## Retrieval presets

The panel's preset (`maxContextMessages`, sent as 4 / 10 / 16) picks a budget
in `memory/context.py`:

| Preset | Memory section | Facts / entities from the template | Last-conversation messages |
| --- | --- | --- | --- |
| lean | 1,500 chars | 10 / 5 | 4 |
| balanced | 3,000 chars | 20 / 10 | 10 |
| rich | 6,000 chars | 40 / 20 | 16 |

`minFactRelevance` is applied to the facts of the fallback path (the
cross-encoder's `relevance`). Context templates are shared, one per budget
(`kwami-context-v2-e{edges}-n{entities}`), rather than one per user.

## Continuity between sessions

Every session is a new Zep thread, empty when the greeting is prepared, and
Zep's context for an empty thread has nothing to be relevant to. So:

- `load_last_conversation` finds the most recent earlier thread with anything
  in it (empty reconnect threads are skipped, up to 4) and reads its last
  `maxContextMessages` messages.
- Until this session has an exchange of its own, `get_context` asks Zep for
  context relative to *that* thread, and the prompt gets a short "How your last
  conversation ended" section (at most a third of the budget, most recent lines
  kept). The greeting sees a returning user.
- After the first exchange, and every 3rd after it, the memory block is read
  again for the current thread. On the standard pipeline the instructions are
  updated when the block changed. Realtime and hybrid models are left alone:
  updating their instructions mid-session resends the whole prompt as a turn
  on the Gemini API, so they recall through `recall_memories`.

## Identity

- Zep `user_id` = `kwami_id` from the config message (or telephony bootstrap)
- Typical shape: `kwami_<supabaseUserId>_<kwamiId>`
- Thread id is minted per conversation (`session_{user}_{uuid4}`) unless the
  config supplies `session_id`

Reusing the same `user_id` across config updates is required for recall. A
previous bug created a new `AsyncZep` and a new thread on every `config`
message, so facts were written and never found.

`handle_full_config` therefore **reuses** the live `KwamiMemory` when the new
config targets the same Zep user, and only copies retrieval knobs
(`max_context_messages`, `include_facts`, `min_fact_relevance`). A different
user id creates a new client; the old one is closed by `SessionState`.

## Session timeline

```mermaid
sequenceDiagram
    participant Agent as KwamiAgent
    participant Mem as KwamiMemory
    participant Zep as Zep Cloud

    Agent->>Mem: create_memory (first config only)
    Mem->>Zep: ensure user, thread, ontology, context template
    Note over Agent: on_enter
    par within 6s total
        Agent->>Mem: get_user_name()
        Agent->>Mem: get_context()
    end
    alt budget exceeded
        Agent->>Agent: greet without memory
    else context arrived
        Agent->>Agent: update_instructions + personalized greeting
    end
    loop each turn
        Agent->>Mem: buffer user message
        Agent->>Mem: add_exchange(user + assistant)
    end
```

Budgets (see `constants.Timeouts`):

| Budget | Value | Why |
| --- | --- | --- |
| `ZEP_REQUEST` | 8s | Zep's SDK default is 60s per call |
| `MEMORY_CONTEXT` | 6s | `get_user_name` + `get_context` together, before greeting |
| Greeting fallback | immediate | Product is a voice; silence is worse than a generic hello |

`get_user_name` and `get_context` run concurrently. The greeting reuses the
cached context so it does not pay a second round trip.

## Ontology

Custom types extend Zep defaults (User, Assistant, Preference, Location,
Event, Object, Topic, Organization, Document). Edges carry explicit
source **and** target constraints so the graph does not grow User-centric
orphans.

```mermaid
erDiagram
    User ||--o{ Person : KNOWS
    User ||--o{ Organization : WORKS_AT
    User ||--o{ Location : LIVES_IN
    User ||--o{ Topic : INTERESTED_IN
    User ||--o{ Project : WORKING_ON
    User ||--o{ Goal : HAS_GOAL
    User ||--o{ Product : USES
    User ||--o{ Preference : PREFERS
    User ||--o{ Task : HAS_TASK
    User ||--o{ Asset : HOLDS
    Person ||--o{ Organization : WORKS_AT
    Person ||--o{ Location : LIVES_IN
    Person ||--o{ Project : WORKING_ON

    Person {
        string relationship
    }
    Project {
        string status
    }
    Product {
        string category
    }
    Goal {
        string timeframe
    }
    Procedure {
        string detail
    }
    Task {
        string due
        string status
    }
    Asset {
        string symbol
        string position
    }
```

`set_ontology` replaces the user's ontology, so it is applied once per
version: the agent stamps `kwami_ontology: "<version>"` on the user's Zep
metadata after applying the defaults, skips stamped users, and never touches a
user marked `"custom"` (an ontology set from the memory panel). Bump
`ONTOLOGY_VERSION` in `memory/manager.py` when the default types change.

The user's name is kept on their Zep record (`first_name`) once learned, so Zep
names the user's own node after it and the next session reads it without a
graph search.

## Agent tools

| Tool | Purpose |
| --- | --- |
| `remember_fact` | Write an explicit fact into the graph |
| `recall_memories` | Search threads / graph for a topic |
| `get_memory_status` | Initialized?, user id, cached name |

Search-derived facts are written in the background so a Zep hiccup does not
block the spoken answer.

## Billing

Zep retrieval is metered only when it returned content. Empty or failed
lookups do not increment the usage tracker. See [Billing](./billing.md).

## Failure modes

| Failure | Behaviour |
| --- | --- |
| No `ZEP_API_KEY` | Memory stays disabled; no tools talk to Zep |
| Init / ontology error | Logged; session continues without memory |
| Context timeout | Greeting without personalization |
| Write error mid-session | Logged; conversation continues |
| Config targets a new user | New client; old client closed |
| Config targets same user | Live client reused |
