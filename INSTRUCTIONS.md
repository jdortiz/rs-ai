# AI Architectures in Rust — Workshop Instructions

In this workshop you will build three small AI applications in Rust, each one a step further from a plain chatbot towards an
agent:

| Stage | Package       | What you build                                              | Main crates               |
|-------|---------------|-------------------------------------------------------------|---------------------------|
| 1     | `chat-repl`   | An interactive chat (REPL) with conversation memory         | `llm`, `rustyline`        |
| 2a    | `rag`         | Retrieval-Augmented Generation with an in-memory vector store | `rig-core`                |
| 2b    | `rag`         | The same RAG, simplified with an agent                      | `rig-agent`               |
| 3a    | `agent-tools` | An agent that uses tools through a hand-written agent loop  | `rig-core`, `serde_json`  |
| 3b    | `agent-tools` | *(Bonus)* The same agent using native tool calling          | `rig-agent`               |

All three packages use [`tracing`](https://docs.rs/tracing) so you can *see* what the application does: every user turn
gets a span, and model calls, retrieval and tool usage are recorded inside it.

---

## 0. Before you start

### 0.1 Branches

| Branch     | Contents                                                     |
|------------|--------------------------------------------------------------|
| `main`     | **Starting point.** You work here.                           |
| `solution` | Finished version of every stage, one commit per stage.       |

```shell
git clone https://github.com/jdortiz/rs-ai.git
cd rs-ai
```

Each stage of the solution is also tagged, so you can look at (or jump to) the code at the end of any stage:

| Tag                   | End of stage                          |
|-----------------------|---------------------------------------|
| `stage-1-chat-repl`   | 1 — Chat REPL                         |
| `stage-2a-rag`        | 2a — RAG                              |
| `stage-2b-rag-agent`  | 2b — RAG with an agent                |
| `stage-3a-agent-tools`| 3a — Agent with tools                 |
| `stage-3b-rig-tools`  | 3b — Native tool calling (bonus)      |

If you get stuck, you can compare your code with the solution without leaving your branch:

```shell
git log --oneline main..solution                   # one commit per stage
git diff main solution -- chat-repl/src/main.rs    # whole solution for a package
git show stage-2a-rag:rag/src/main.rs              # a file at the end of a stage
```

And if you fall behind, you can replace a package with the version at the end of a stage and carry on from there
(this **overwrites** your changes to that package):

```shell
git checkout stage-2a-rag -- rag/
```

### 0.2 Toolchain

- Rust **1.85 or newer** (the workspace uses edition 2024). Check with `rustc --version`.
- The workspace enables `clippy::pedantic` warnings. Run `cargo clippy -p <package>` from time to time to get hints.

### 0.3 A model provider

Every package can talk to **Ollama** (local, default) or **OpenAI** (remote, enabled with the `open-ai-api` feature).

**Option A — Ollama (default)**

1. Install [Ollama](https://ollama.com) and make sure it is running (`ollama list` should work).
2. Pull the models:
   ```shell
   ollama pull devstral
   ollama pull nomic-embed-text
   ```
3. The code uses a variant of `devstral` with a large context window, named `devstral_128k_ctx`. Create it from a
   `Modelfile`:
   ```
   FROM devstral
   PARAMETER num_ctx 131072
   ```
   ```shell
   ollama create devstral_128k_ctx -f Modelfile
   ```
   > `devstral` is a ~14 GB download. Do this **before** the workshop. If your machine can't run it, use OpenAI.

**Option B — OpenAI**

1. Put your key in the `.env` file (see below): `OPENAI_API_KEY=sk-...`
2. Add `--features open-ai-api` to every `cargo run` command in this guide, e.g.
   `cargo r -p chat-repl --features open-ai-api`.

### 0.4 The `.env` file

All programs load configuration from a `.env` file in the repository root and **panic if it doesn't exist**. It is ignored
by git, so create it yourself (it may be empty when using Ollama):

```shell
touch .env
```

### 0.5 Check that everything builds

```shell
cargo build
```

The first build downloads and compiles all dependencies; better to do it before the workshop starts.

### 0.6 Conventions used in this guide

- Snippets show the code you need to **add or change**, not the complete file.
- Rust is strict about `use` statements. Each stage lists the imports you'll need at the beginning; alternatively, let
  your editor/rust-analyzer add them as you go.
- Run commands from the repository root. `cargo r` is short for `cargo run`.
- `RUST_LOG=<crate>=debug` shows the tracing output of the package. Note that the crate name uses `_` where the package
  name uses `-` (`chat-repl` → `chat_repl`).

---

## Stage 1 — Chat REPL (`chat-repl`)

### What you start with

`chat-repl/src/main.rs` is based on the example from the [`llm` crate](https://crates.io/crates/llm), with some changes:

1. Errors are handled with `anyhow`.
2. Configuration is read from `.env` with `dotenvy`.
3. `tracing` and `tracing-subscriber` show what happens during startup.
4. The `open-ai-api` feature selects OpenAI; without it, Ollama is used.
5. Model initialization lives in `build_llm()` (one version per provider) and the system prompt describes a Rust
   expert.

Try it first:

```shell
RUST_LOG=chat_repl=debug cargo r -p chat-repl
```

It sends a hard-coded conversation once and exits. Your job is to turn it into an **interactive chat that remembers the
conversation**, and to trace every turn.

### Imports you'll need

```rust
use rustyline::{DefaultEditor, error::ReadlineError};
use tracing::{Instrument, debug, error, field, info, info_span, warn};
```

### 1.1 Extract the interaction with the model into a function

1. Declare the function that will handle each interaction of the user with the chat. It receives the model, the
   conversation so far, the turn number and the length of the user input (the last two are for tracing).
   ```rust
   async fn chat_interaction(
       llm: &dyn LLMProvider,
       messages: &mut Vec<ChatMessage>,
       turn: u32,
       input_chars: usize,
   ) {
   }
   ```
2. Move the whole `match llm.chat(&messages).await { ... }` block from `main` into the body of that function.
   Inside the function `messages` is already a reference, so the call becomes `llm.chat(messages)`.
3. Make `messages` mutable in `main` and **remove the third message** (the hard-coded question about the macro); from now
   on the user types the questions.
   ```rust
   let mut messages = vec![
       ChatMessage::user().content("Hello. Call me Jorge.").build(),
       ChatMessage::assistant()
           .content("Hello, Jorge! How can I help you today?")
           .build(),
   ];
   ```

### 1.2 Read user input in a loop

4. Create a line editor with [`rustyline`](https://docs.rs/rustyline). It gives you history, arrow keys and Ctrl-C/Ctrl-D
   handling for free.
   ```rust
   let mut rl = DefaultEditor::new()?;
   ```
5. Greet the user, initialize the turn counter and start a loop. Keep `Ok(())` after the loop, at the end of `main`.
   ```rust
   println!("Greetings Professor Falken!\n\nHow are you feeling today?");
   let mut turn = 0_u32;

   loop {
   }
   ```
6. Inside the loop, ask for input with a prompt:
   ```rust
   let readline = rl.readline("WOPR > ");
   ```
7. And process the result with pattern matching:
   ```rust
   match readline {
   }
   ```
8. **Happy path:** the input contains a line. Ignore empty lines.
   ```rust
   Ok(line) => {
       if line.trim().is_empty() {
           warn!(reason = "empty_prompt", "Ignoring empty prompt");
       } else {
       }
   }
   ```
9. In the `else` branch, count the turn, add the user message to the conversation, and then invoke `chat_interaction()`:
   ```rust
   turn += 1;
   let input_chars = line.chars().count();
   messages.push(ChatMessage::user().content(line).build());
   chat_interaction(&*llm, &mut messages, turn, input_chars).await;
   ```
   > `llm` is a `Box<dyn LLMProvider>`; `&*llm` borrows the provider inside the box as `&dyn LLMProvider`.
10. **Not-so-happy paths.** The user pressed Ctrl-C:
    ```rust
    Err(ReadlineError::Interrupted) => {
        warn!(reason = "interrupt", "Interactive session interrupted");
        break;
    }
    ```
11. The user pressed Ctrl-D:
    ```rust
    Err(ReadlineError::Eof) => {
        warn!(reason = "eof", "Interactive session finished");
        break;
    }
    ```
12. Any other error:
    ```rust
    Err(err) => {
        error!(error = %err, "Reading interactive input failed");
        break;
    }
    ```

At this point the program should compile. Before running it, let's make each turn observable.

### 1.3 Trace each turn

13. At the beginning of `chat_interaction()`, create a span for the turn. Fields whose value is not known yet are
    declared as `field::Empty` and recorded later.
    ```rust
    let turn_span = info_span!(
        "turn",
        turn,
        input_chars,
        messages_before = messages.len(),
        messages_after = field::Empty,
        response_chars = field::Empty,
        prompt_tokens = field::Empty,
        completion_tokens = field::Empty,
        outcome = field::Empty,
    );
    ```
14. Call the model *before* the `match`, instrumenting the future with the span so that everything logged by the
    provider during the request is attached to this turn:
    ```rust
    let response_result = llm.chat(messages).instrument(turn_span.clone()).await;

    match response_result {
        // ...existing Ok/Err arms...
    }
    ```
15. In the `Ok` arm, when the response has text (`if let Some(text)`), record it in the span and **add the answer to the
    conversation**. This is what gives the chat its memory.
    ```rust
    turn_span.record("response_chars", text.chars().count());
    turn_span.record("outcome", "ok");
    messages.push(ChatMessage::assistant().content(text).build());
    ```
16. Replace the `info!` about token usage so it is recorded in, and emitted inside, the span:
    ```rust
    turn_span.record("prompt_tokens", usage.prompt_tokens);
    turn_span.record("completion_tokens", usage.completion_tokens);
    turn_span.in_scope(|| {
        info!(
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            "Chat token usage"
        );
    });
    ```
17. The same for the case without usage information:
    ```rust
    turn_span.in_scope(|| info!("Chat provider returned no usage information"));
    ```
18. And for the error arm:
    ```rust
    Err(err) => {
        turn_span.record("outcome", "provider_error");
        turn_span.in_scope(|| error!(error = %err, "Chat request failed"));
    }
    ```
19. Finally, after the `match`, record how the conversation grew and close the turn:
    ```rust
    turn_span.record("messages_after", messages.len());
    turn_span.in_scope(|| {
        debug!("Chat turn completed");
    });
    ```

### 1.4 Try it

```shell
RUST_LOG=chat_repl=debug cargo r -p chat-repl
```

First ask:

```
Write a declarative macro to create a HashMap using any number of tuples.
```

Then:

```
Can you write a couple of tests for it?
```

The second question only makes sense if the model remembers the first one. Look at `messages_before`/`messages_after` in
the logs to see the conversation grow. Exit with Ctrl-D.

**Solution:** `git diff main stage-1-chat-repl -- chat-repl/`

---

## Stage 2a — RAG (`rag`)

### What you start with

`rag/src/main.rs` is similar to the chat REPL, but:

1. It uses [`rig`](https://docs.rs/rig-core) (`rig-core`) instead of `llm`.
2. It sends single prompts (a completion with a preamble) instead of a whole conversation.
3. It is already instrumented, and the turn span has a `retrieval` field set to `"none"`.

LLMs only know what they saw during training. Ask the current version about a recent Rust release and it will either
admit it doesn't know or make something up. **RAG** (Retrieval-Augmented Generation) fixes this by looking up relevant
documents and passing them to the model together with the question.

How do we find "relevant" documents? With **vector search**: an *embedding model* turns each text into a vector of
numbers such that texts with similar meaning end up close to each other. We compute the vectors of our documents once,
store them in a *vector store*, and for each question we compute its vector and fetch the nearest documents.

### Imports you'll need

```rust
use std::collections::HashMap;

#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai::{self, TEXT_EMBEDDING_ADA_002}; // replaces the existing openai import
use rig_core::{
    Embed,
    completion::{CompletionRequest, Document},
    embeddings::EmbeddingsBuilder,
    vector_store::{VectorSearchRequest, VectorStoreIndex, in_memory_store::InMemoryVectorStore},
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;
```

### 2a.1 Define and ingest the documents

1. Define the type of the documents. The field used to compute the embedding is marked with `#[embed]`; the rest is
   stored alongside it.
   ```rust
   struct Note {
       id: Ulid,
       title: String,
       #[embed]
       content: String,
   }
   ```
2. Derive the traits it needs: `Embed` to compute the embedding, `Serialize` to store it in the vector store, and
   `Deserialize` to get it back from search results.
   ```rust
   #[derive(Deserialize, Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
   ```
3. Write a function that simulates the ingestion of the documentation. In a real application this would read files, a
   database, a wiki…
   ```rust
   fn fetch_notes() -> Vec<Note> {
   }
   ```
4. Return a first hard-coded note (taken from the Rust 1.93 release notes):
   ```rust
   vec![
       Note {
           id: Ulid::new(),
           title: "Allow the global allocator to use thread-local storage".to_string(),
           content: "Rust 1.93 adjusts the internals of the standard library to permit global allocators written in Rust to use std's `thread_local!` and `std::thread::current` without re-entrancy concerns by using the system allocator instead.".to_string(),
       },
   ]
   ```
5. And another one, inside the same `vec!`:
   ```rust
   Note {
       id: Ulid::new(),
       title: "Validate input to `#[macro_export]`".to_string(),
       content: "Over the past few releases, many changes were made to the way built-in attributes are processed in the compiler. This should greatly improve the error messages and warnings Rust gives for built-in attributes and especially make these diagnostics more consistent among all of the over 100 built-in attributes.

   To give a small example, in this release specifically, Rust became stricter in checking what arguments are allowed to macro_export by upgrading that check to a 'deny-by-default lint' that will be reported in dependencies.".to_string(),
   },
   ```
6. Feel free to add your own notes later using this template:
   ```rust
   /*
   Note {
       id: Ulid::new(),
       title: "".to_string(),
       content: "".to_string(),
   },
   */
   ```

### 2a.2 Configure an embedding model

Embeddings are computed by a specific kind of model, not by the LLM that generates the answers. Change the startup code
so that it creates **both** models from the same provider.

7. For Ollama (the `#[cfg(not(feature = "open-ai-api"))]` block):
   ```rust
   let (gen_model, emb_model) = startup_span.in_scope(|| {
       info!(
           provider = "ollama",
           embedding_model = "nomic-embed-text",
           generation_model = "devstral_128k_ctx:latest",
           "Configuring RAG provider"
       );
       let provider = ollama::Ollama::new();

       (
           provider.native_completion("devstral_128k_ctx:latest"),
           provider.embedding("nomic-embed-text", None),
       )
   });
   ```
8. For OpenAI (the `#[cfg(feature = "open-ai-api")]` block):
   ```rust
   let (gen_model, emb_model) = startup_span.in_scope(|| {
       info!(
           provider = "openai",
           embedding_model = "text-embedding-ada-002",
           generation_model = "gpt-4o-mini",
           "Configuring RAG provider"
       );
       let provider = openai::OpenAI::from_env().expect("Unable to create OpenAI client instance");

       (
           provider.chat(openai::GPT_4O_MINI),
           provider.embedding(TEXT_EMBEDDING_ADA_002, None),
       )
   });
   ```

### 2a.3 Build the vector store and the index

Add the following after creating the models and **before** `drop(startup_span);`, so the work is traced as part of
startup.

9. Fetch the documents:
   ```rust
   let docs = fetch_notes();
   let document_count = docs.len();
   ```
10. Compute their embeddings. This calls the embedding model once per document, so it is async:
    ```rust
    let embeddings = EmbeddingsBuilder::new(emb_model.clone())
        .documents(docs)?
        .build()
        .instrument(info_span!(
            parent: &startup_span,
            "embedding_build",
            document_count
        ))
        .await?;
    ```
11. Store documents and embeddings in a vector store — an in-memory one for this example (rig has integrations for
    Qdrant, LanceDB, MongoDB, etc.):
    ```rust
    let vector_store = {
        let _span = info_span!(
            parent: &startup_span,
            "vector_store_build",
            document_count
        )
        .entered();
        let vector_store = InMemoryVectorStore::from_documents(embeddings);
        debug!("Vector store built");
        vector_store
    };
    ```
12. Create an index on the store to search it. The index needs the embedding model to turn each query into a vector:
    ```rust
    let index = {
        let _span = info_span!(
            parent: &startup_span,
            "index_build",
            index_type = "in_memory"
        )
        .entered();
        let index = vector_store.index(emb_model);
        debug!("Indexes created");
        index
    };
    ```

### 2a.4 Use the retrieved documents in each turn

13. Change the `retrieval` field of the `turn_span`:
    ```rust
    retrieval = "rig.dynamic_context",
    ```
14. After creating the span, look for the document that is semantically closest to the user's prompt (`samples(1)` →
    top 1):
    ```rust
    let results = index
        .top_n::<Note>(
            VectorSearchRequest::builder()
                .query(line.clone())
                .samples(1)
                .build(),
        )
        .instrument(turn_span.clone())
        .await?;
    ```
15. Convert the hits into the `Document`s that a completion request understands:
    ```rust
    let documents = results.into_iter().map(|result| Document {
        id: result.id,
        text: result.document.content,
        additional_props: HashMap::from([(
            String::from("title"),
            result.document.title,
        )]),
    });
    ```
16. And attach them to the completion request:
    ```rust
    let response_result = gen_model
        .call(
            CompletionRequest::new(line)
                .preamble(AGENT_RULES)
                .documents(documents),
        )
        .instrument(turn_span.clone())
        .await;
    ```

### 2a.5 Try it

```shell
RUST_LOG=rag=debug cargo r -p rag
```

```
Can I use the thread-local storage from the global allocator?
```

The answer should now be based on the note. To see the difference, compare it with the answer you get from the
starting version (`git stash`, run, `git stash pop`).

**Solution:** `git diff main stage-2a-rag -- rag/`

---

## Stage 2b — Simplify the RAG with an agent (`rag`)

Searching, converting the results and attaching them to the request is a common pattern. The
[`rig-agent`](https://docs.rs/rig-agent) crate packages it (and much more) into an `Agent`.

1. Add the dependency:
   ```shell
   cargo add -p rag rig-agent@0.45
   ```
   and the import:
   ```rust
   use rig_agent::AgentBuilder;
   ```
2. After building the index and before `drop(startup_span);`, build an agent. `dynamic_context(1, index)` tells it to
   fetch the closest document from the index for every prompt — exactly what you did by hand in steps 14–16.
   ```rust
   let agent = {
       let _span = info_span!(
           parent: &startup_span,
           "agent_build",
       )
       .entered();
       let agent = AgentBuilder::new(gen_model)
           .preamble(AGENT_RULES)
           .dynamic_context(1, index)
           .build();
       debug!("Agent built");
       agent
   };
   ```
3. Change the `retrieval` field of the `turn_span`:
   ```rust
   retrieval = "rig.agent",
   ```
4. Remove the two retrieval steps (`let results = ...` and `let documents = ...`).
5. Prompt the agent instead of calling the model:
   ```rust
   let response_result = agent
       .prompt(line)
       .into_future()
       .instrument(turn_span.clone())
       .await;
   ```
   > `prompt()` returns a request builder that implements `IntoFuture`; `.into_future()` turns it into a real future so
   > we can call `.instrument()` on it.
6. The agent's response is read with `output()` instead of `text()`:
   ```rust
   turn_span.record("response_chars", response.output().chars().count());
   turn_span.in_scope(|| debug!("Response received"));
   println!("{}", response.output());
   ```
7. Clean up the imports that are no longer used (`HashMap`, `CompletionRequest`, `Document`, `VectorSearchRequest`,
   `VectorStoreIndex`). `cargo clippy -p rag` will point them out.
8. Run and test again with the same prompt:
   ```shell
   RUST_LOG=rag=debug cargo r -p rag
   ```
   ```
   Can I use the thread-local storage from the global allocator?
   ```

**Solution:** `git diff stage-2a-rag stage-2b-rag-agent -- rag/`

---

## Stage 3a — An agent that uses tools (`agent-tools`)

### What you start with

`agent-tools/src/main.rs` evolves from the RAG code, without documents or embeddings. It:

- keeps a chat history (so it remembers the conversation),
- uses a `DynModel<Completion>` so the rest of the code doesn't care which provider is behind it,
- has three tool functions — `list_files`, `read_file` and `terminate` — and two helpers to parse the model's answer,
  **commented out** at the bottom of the file.

Try the original version first:

```shell
RUST_LOG=agent_tools=debug cargo r -p agent-tools
```

```
Could you help me improve the Cargo configuration of this Rust project?
```

The model can only *tell* you what to do; it can't look at your project. We'll let it **act**: the model will reply with
an *action* (a tool name and its arguments), our code will execute it and feed the result back, and this repeats until
the model decides the task is done. That loop is what makes it an *agent*.

### Imports you'll need

```rust
use std::{collections::HashMap, fs, path::Path};

use anyhow::{anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{Instrument, Span, debug, debug_span, error, field, info, info_span, warn};
```

### 3a.1 Describe the actions

1. Define the type an action is deserialized into:
   ```rust
   #[derive(Deserialize)]
   struct Action {
       tool_name: String,
       args: HashMap<String, String>,
   }
   ```
2. Uncomment the tools and helpers at the bottom of the file (remove the `/*` and `*/`). Take a minute to read them:
   - Each tool returns a `serde_json::Value` that will be sent back to the model.
   - `extract_action_block()` finds a fenced block marked as `action` in the model's markdown answer, and `parse_action()`
     deserializes it into an `Action`.
3. Replace the system prompt (`AGENT_RULES`) to tell the model which tools exist and **how** to ask for them:
   ````text
   You are an AI agent expert in software engineering, that can perform tasks by using available tools.  You use Rust as your main programming language.

   Available tools:
   - list_files: arguments ()  returns [str]: List all files in the current directory.
   - read_file: arguments (file_name: str), returns str: Read the content of a file.
   - terminate: arguments (message: str) returns (): End the agent loop and print a summary to the user.

   If a user asks about files, list them before reading.

   Every response MUST have an action, even if it is the terminate one, and an explanation.
   Use this format, including the back quotes, for the action:
   ```action
   {
       "tool_name": "insert tool_name",
       "args": {...fill in any required arguments here...}
   }
   ```
   ````

### 3a.2 Write the agent loop

4. The agent needs its own internal loop, in which it uses the tools to work out the answer to the user's prompt.
   Declare the function:
   ```rust
   /// Makes the agent iterate to achieve the goal.
   async fn perform_agent_loop(
       gen_model: &DynModel<Completion>,
       user_input: &str,
       chat_history: &mut Vec<Message>,
       turn_span: &Span,
   ) -> anyhow::Result<String> {
   }
   ```
5. Emit an event to signal the start of the work and loop for a bounded number of iterations — a model that never
   terminates must not loop forever (or burn your API credits). `agent_input` is what we send to the model on each
   iteration: first the user's request, later the tool results.
   ```rust
   turn_span.in_scope(|| {
       info!("Agent processing user request.");
   });

   const MAX_ITERATIONS: u32 = 20;
   let mut agent_input = String::from(user_input);

   for iteration in 1..=MAX_ITERATIONS {
   }
   ```
6. Inside the loop, create a child span for the iteration and emit an event:
   ```rust
   let iteration_span = debug_span!(
       parent: turn_span,
       "iteration",
       iteration,
       input_chars = agent_input.chars().count(),
       history_messages = chat_history.len(),
       outcome = field::Empty,
   );
   iteration_span.in_scope(|| {
       debug!(
           task = user_input,
           agent_input,
           ?chat_history,
           "Starting iteration",
       );
   });
   ```
7. **Move** the call to the model from `main` into the loop, sending `agent_input` and instrumenting it with the
   iteration span:
   ```rust
   let response_result = gen_model
       .call(
           CompletionRequest::new(&agent_input)
               .preamble(AGENT_RULES)
               .messages(chat_history.iter().cloned()),
       )
       .instrument(iteration_span.clone())
       .await;
   ```
8. Back in `main`, replace the code you moved and the whole `match response_result { ... }` with a call to the new
   function. Notice two changes: the history is now updated inside the agent loop, and an error no longer ends the
   program (`return Err(...)` is gone) — the user can try again.
   ```rust
   match perform_agent_loop(&gen_model, &line, &mut chat_history, &turn_span).await
   {
       Ok(agent_response) => {
           turn_span.record("outcome", "ok");
           turn_span.record("response_chars", agent_response.len());
           turn_span.in_scope(|| debug!("Agent response received"));
           println!("{}", agent_response);
       }
       Err(err) => {
           turn_span.record("outcome", "agent_error");
           turn_span.in_scope(|| {
               error!(turn, error = %err, "Agent prompt failed");
           });
       }
   }
   ```
9. Back in the loop of `perform_agent_loop()`, handle the result of the model call. The error arm is similar to what we
   had:
   ```rust
   match response_result {
       Ok(response) => {
       }
       Err(err) => {
           iteration_span.record("outcome", "model_error");
           turn_span.record("outcome", "agent_error");
           iteration_span.in_scope(|| {
               error!(error = %err, "Agent prompt failed");
           });
           return Err(err.into());
       }
   }
   ```
10. In the `Ok` arm, try to extract an action from the response. If there is none, the model answered directly, and that
    answer completes the turn:
    ```rust
    let response_text = response.text().to_owned();

    let action_request = match iteration_span.in_scope(|| parse_action(&response_text))
    {
        Ok(action_request) => {
            iteration_span.in_scope(|| {
                debug!(
                    ?action_request.tool_name,
                    ?action_request.args, "Agent iteration"
                )
            });
            action_request
        }
        Err(err) => {
            iteration_span.record("outcome", "completed");
            iteration_span.in_scope(|| {
                warn!(error = %err, "No action found; completing turn");
            });
            return Ok(response_text);
        }
    };
    ```
11. Dispatch the action to the right tool:
    ```rust
    let is_termination = action_request.tool_name == "terminate";
    let result = iteration_span.in_scope(|| match action_request.tool_name.as_str() {
        "list_files" => {
        }
        "read_file" => {
        }
        "terminate" => {
        }
        _ => {
        }
    });
    ```
12. List files:
    ```rust
    debug!(action = "list_files", "Executing action");
    list_files()
    ```
13. Read a file. The model may forget the argument, so we return an error *to the model* instead of failing:
    ```rust
    debug!(action = "read_file", "Executing action");
    if let Some(file_name) = action_request.args.get("file_name") {
        read_file(file_name)
    } else {
        json!({"error": "Missing file_name argument."})
    }
    ```
14. Terminate:
    ```rust
    debug!(action = "terminate", "Executing action");
    terminate(
        action_request
            .args
            .get("message")
            .map(String::as_str)
            .unwrap_or("That's all folks!"),
    )
    ```
15. An unknown tool (models do hallucinate tool names):
    ```rust
    debug!("Requested unexpected action");
    json!({"error": "unknown tool"})
    ```
16. After the dispatch, either finish (on `terminate`) or record the exchange in the history and send the tool result
    to the model in the next iteration:
    ```rust
    if is_termination {
        iteration_span.record("outcome", "terminated");
        return Ok(action_request
                  .args
                  .get("message")
                  .cloned()
                  .unwrap_or_else(|| "That's all folks!".to_owned()));
    }

    // Preserve the action exchange in history, then feed the tool
    // outcome into the next model request.
    chat_history.push(Message::user(&agent_input));
    chat_history.push(Message::assistant(response_text));
    agent_input = result.to_string();
    iteration_span.record("outcome", "tool_requested");
    ```
17. After the `for` loop (i.e. we ran out of iterations), return an error:
    ```rust
    turn_span.record("outcome", "iteration_limit");
    Err(anyhow!(
        "Agent exceeded the maximum of {MAX_ITERATIONS} iterations"
    ))
    ```

### 3a.3 Try it

Run the new version:

```shell
RUST_LOG=agent_tools=debug cargo r -p agent-tools
```

And use the same prompt as before:

```
Could you help me improve the Cargo configuration of this Rust project?
```

Follow the `iteration` spans in the logs: you should see the agent list the files, read `Cargo.toml` (maybe several of
them), and finally `terminate` with its suggestions — now based on your actual project.

> The tools run relative to the **current directory**, so run the program from the repository root.

**Solution:** `git diff main stage-3a-agent-tools -- agent-tools/`

---

## Stage 3b (Bonus) — Native tool calling with `rig-agent` (`agent-tools`)

Writing the tool protocol in the prompt and parsing markdown works with any model, but modern models support **tool
calling** natively: the tools are described with a JSON schema, and the model returns structured tool calls. `rig-agent`
also runs the agent loop for you.

Outline of the changes (see the solution for the full code):

1. Add `rig-agent = { workspace = true }` to `agent-tools/Cargo.toml` (and `rig-agent = "0.45"` to
   `[workspace.dependencies]` in the root `Cargo.toml`).
2. Simplify `AGENT_RULES`: no tool list and no action format, just *when* to use the tools.
3. Implement the `rig_agent::tool::Tool` trait for `ListFiles`, `ReadFile` and `Terminate`: a `NAME`, typed `Args`
   (deserializable structs), `Output`, `Error`, a `description()`, the JSON schema of the `parameters()`, and an async
   `call()`.
4. Make `list_files`, `read_file` and `terminate` return plain Rust types (`io::Result<Vec<String>>`,
   `io::Result<String>`, `String`) — serialization is now rig's job.
5. Build an `Agent` at startup with `.preamble(AGENT_RULES)`, `.default_max_turns(20)` and `.tool(...)` for each tool.
6. Reduce `perform_agent_loop()` to `agent.chat(user_input.to_owned(), chat_history)` instrumented with the turn span,
   returning `response.output()`.
7. Delete `Action`, `parse_action()` and `extract_action_block()`.

Run it with the same prompt and compare the logs and the result with stage 3a.

**Solution:** `git diff stage-3a-agent-tools stage-3b-rig-tools -- agent-tools/ Cargo.toml`

---

## Running the final version

```shell
git switch solution
RUST_LOG=chat_repl=debug   cargo r -p chat-repl
RUST_LOG=rag=debug         cargo r -p rag
RUST_LOG=agent_tools=debug cargo r -p agent-tools
```

(Add `--features open-ai-api` to use OpenAI.)

## Troubleshooting

| Symptom                                                  | Fix                                                                       |
|----------------------------------------------------------|---------------------------------------------------------------------------|
| `Unable to load configuration from .env`                 | `touch .env` in the repository root.                                      |
| Connection refused / model not found (Ollama)            | Start Ollama and check `ollama list` shows `devstral_128k_ctx` and `nomic-embed-text`. |
| `Unable to create OpenAI client instance`                | Set `OPENAI_API_KEY` in `.env` and run with `--features open-ai-api`.     |
| No logs at all                                           | Set `RUST_LOG`, and use `_` in the crate name (`chat_repl`, `agent_tools`). |
| Agent never calls tools / answers without an action      | Smaller models don't always follow the format; retry, or try another model. |
| `read_file` errors                                       | Run from the repository root; the tools use paths relative to the current directory. |
