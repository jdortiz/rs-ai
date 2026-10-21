use rig_core::completion::CompletionRequest;
#[cfg(not(feature = "open-ai-api"))]
use rig_core::providers::ollama;
#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai;
use rustyline::{DefaultEditor, error::ReadlineError};
use tracing::{Instrument, debug, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say \"I don't know.\"
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let startup_span = info_span!("startup");
    #[cfg(not(feature = "open-ai-api"))]
    let gen_model = startup_span.in_scope(|| {
        info!(
            provider = "ollama",
            generation_model = "devstral_128k_ctx:latest",
            "Configuring RAG provider"
        );
        ollama::Ollama::new().native_completion("devstral_128k_ctx:latest")
    });
    #[cfg(feature = "open-ai-api")]
    let gen_model = startup_span.in_scope(|| {
        info!(
            provider = "openai",
            generation_model = "gpt-4o-mini",
            "Configuring RAG provider"
        );
        openai::OpenAI::from_env()
            .expect("Unable to create OpenAI client instance")
            .chat(openai::GPT_4O_MINI)
    });
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
                        retrieval = "none",
                        outcome = field::Empty,
                        response_chars = field::Empty,
                    );

                    let response_result = gen_model
                        .call(CompletionRequest::new(line).preamble(AGENT_RULES))
                        .instrument(turn_span.clone())
                        .await;
                    match response_result {
                        Ok(response) => {
                            turn_span.record("outcome", "ok");
                            turn_span.record("response_chars", response.text().chars().count());
                            turn_span.in_scope(|| debug!("Response received"));
                            println!("{}", response.text());
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
