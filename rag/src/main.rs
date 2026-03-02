use log::{error, info, warn};
use rig::{
    Embed,
    client::{CompletionClient, EmbeddingsClient},
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    vector_store::in_memory_store::InMemoryVectorStore,
};
#[cfg(not(feature = "open-ai-api"))]
use rig::{client::Nothing, providers::ollama};
#[cfg(feature = "open-ai-api")]
use rig::{
    client::ProviderClient,
    providers::openai::{self, TEXT_EMBEDDING_ADA_002},
};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::Serialize;
use ulid::Ulid;

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say \"I don't know.\"
"#;

// A vector search will be performed on `content` field.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct Note {
    id: Ulid,
    title: String,
    #[embed]
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    env_logger::init();

    #[cfg(not(feature = "open-ai-api"))]
    info!("Using =Ollama=");
    #[cfg(not(feature = "open-ai-api"))]
    let client: ollama::Client = ollama::Client::new(Nothing)?;
    #[cfg(feature = "open-ai-api")]
    let client = openai::Client::from_env();

    #[cfg(not(feature = "open-ai-api"))]
    let emb_client = client.embedding_model("nomic-embed-text");
    #[cfg(feature = "open-ai-api")]
    let emb_client = client.embedding_model(TEXT_EMBEDDING_ADA_002);

    info!("Computing embeddings for docs");
    let docs = fetch_notes();
    let embeddings = EmbeddingsBuilder::new(emb_client.clone())
        .documents(docs)?
        .build()
        .await?;
    info!("Storing docs wich embeddings");
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    info!("Computing docs indices");
    let index = vector_store.index(emb_client);

    #[cfg(not(feature = "open-ai-api"))]
    let agent = client
        .agent("qwen3-coder:30b") // There are constants like `ollama::LLAMA3_2`, but a string can be used
        .preamble(AGENT_RULES)
        .dynamic_context(1, index)
        .build();
    #[cfg(feature = "open-ai-api")]
    let agent = client
        .agent(openai::GPT_4O_MINI)
        .preamble(AGENT_RULES)
        .dynamic_context(1, index)
        .build();

    let mut rl = DefaultEditor::new()?;
    println!("Greetings Professor Falken!\n\nHow are you feeling today?");
    // Interaction user -- agent
    loop {
        let readline = rl.readline("WOPR > ");

        match readline {
            Ok(line) => {
                if !line.trim().is_empty() {
                    let response = agent.prompt(line).await?;
                    println!("{response}");
                } else {
                    warn!("Empty prompt.");
                }
            }
            Err(ReadlineError::Interrupted) => {
                warn!("Ctrl-C Interrupted");
                break;
            }
            Err(ReadlineError::Eof) => {
                warn!("Ctrl-D Finished");
                break;
            }
            Err(err) => {
                error!("Error: {err:?}");
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
