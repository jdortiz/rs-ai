use log::{error, info, warn};
use rig::{client::CompletionClient, completion::Prompt};
#[cfg(not(feature = "open-ai-api"))]
use rig::{client::Nothing, providers::ollama};
#[cfg(feature = "open-ai-api")]
use rig::{client::ProviderClient, providers::openai};
use rustyline::{DefaultEditor, error::ReadlineError};

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say \"I don't know.\"
"#;

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
    let agent = client
        .agent("qwen3-coder:30b") // There are constants like `ollama::LLAMA3_2`, but a string can be used
        .preamble(AGENT_RULES)
        .build();
    #[cfg(feature = "open-ai-api")]
    let agent = client
        .agent(openai::GPT_4O_MINI)
        .preamble(AGENT_RULES)
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
