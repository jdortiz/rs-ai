use rig_agent::AgentBuilder;
#[cfg(not(feature = "open-ai-api"))]
use rig_core::providers::ollama;
#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai::{self, TEXT_EMBEDDING_ADA_002};
use rig_core::{
    Embed, embeddings::EmbeddingsBuilder, vector_store::in_memory_store::InMemoryVectorStore,
};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::{Deserialize, Serialize};
use tracing::{Instrument, debug, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;
use ulid::Ulid;

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say \"I don't know.\"
"#;

#[derive(Deserialize, Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct Note {
    id: Ulid,
    title: String,
    #[embed]
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let startup_span = info_span!("startup");
    #[cfg(not(feature = "open-ai-api"))]
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
    #[cfg(feature = "open-ai-api")]
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
    let docs = fetch_notes();
    let document_count = docs.len();
    let embeddings = EmbeddingsBuilder::new(emb_model.clone())
        .documents(docs)?
        .build()
        .instrument(info_span!(
            parent: &startup_span,
            "embedding_build",
            document_count
        ))
        .await?;
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

    drop(startup_span);

    let mut rl = DefaultEditor::new()?;
    println!("Greetings Professor Falken!\n\nHow are you feeling today?");
    let mut turn = 0_u32;

    // Interaction user -- agent
    loop {
        let readline = rl.readline("WOPR > ");

        match readline {
            Ok(line) => {
                if line.trim().is_empty() {
                    warn!(reason = "empty_prompt", "Ignoring empty prompt");
                } else {
                    turn += 1;
                    let input_chars = line.chars().count();
                    let turn_span = info_span!(
                        "turn",
                        turn,
                        input_chars,
                        retrieval = "rig.agent",
                        outcome = field::Empty,
                        response_chars = field::Empty,
                    );

                    let response_result = agent
                        .prompt(line)
                        .into_future()
                        .instrument(turn_span.clone())
                        .await;
                    match response_result {
                        Ok(response) => {
                            turn_span.record("outcome", "ok");
                            turn_span.record("response_chars", response.output().chars().count());
                            turn_span.in_scope(|| debug!("Response received"));
                            println!("{}", response.output());
                        }
                        Err(err) => {
                            turn_span.record("outcome", "provider_or_retrieval_error");
                            turn_span.in_scope(|| {
                                error!(turn, error = %err, "RAG prompt failed");
                            });
                            return Err(err.into());
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                warn!(reason = "interrupt", "Interactive session interrupted");
                break;
            }
            Err(ReadlineError::Eof) => {
                warn!(reason = "eof", "Interactive session finished");
                break;
            }
            Err(err) => {
                error!(error = %err, "Reading interactive input failed");
                break;
            }
        }
    }

    Ok(())
}

fn fetch_notes() -> Vec<Note> {
    vec![
        Note {
            id: Ulid::new(),
            title: "Allow the global allocator to use thread-local storage".to_string(),
            content: "Rust 1.93 adjusts the internals of the standard library to permit global allocators written in Rust to use std's `thread_local!` and `std::thread::current` without re-entrancy concerns by using the system allocator instead.".to_string(),
        },
        Note {
            id: Ulid::new(),
            title: "Validate input to `#[macro_export]`".to_string(),
            content: "Over the past few releases, many changes were made to the way built-in attributes are processed in the compiler. This should greatly improve the error messages and warnings Rust gives for built-in attributes and especially make these diagnostics more consistent among all of the over 100 built-in attributes.

To give a small example, in this release specifically, Rust became stricter in checking what arguments are allowed to macro_export by upgrading that check to a 'deny-by-default lint' that will be reported in dependencies.".to_string(),
        },
        /*
    Note {
        id: Ulid::new(),
        title: "".to_string(),
        content: "".to_string(),
        },
         */
    ]
}
