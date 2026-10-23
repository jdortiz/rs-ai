use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};
use rustyline::{DefaultEditor, error::ReadlineError};
use tracing::{Instrument, debug, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let startup_span = info_span!("startup");
    let llm = startup_span.in_scope(|| {
        #[cfg(not(feature = "open-ai-api"))]
        info!(
            provider = "ollama",
            model = "devstral_128k_ctx:latest",
            "Chat provider configured"
        );
        #[cfg(feature = "open-ai-api")]
        info!(
            provider = "openai",
            model = "gpt-4.1-nano",
            "Chat provider configured"
        );

        build_llm()
    });

    // Prepare conversation history with example messages
    let mut messages = vec![
        ChatMessage::user().content("Hello. Call me Jorge.").build(),
        ChatMessage::assistant()
            .content("Hello, Jorge! How can I help you today?")
            .build(),
    ];

    let mut rl = DefaultEditor::new()?;

    println!("Greetings Professor Falken!\n\nHow are you feeling today?");
    let mut turn = 0_u32;

    loop {
        let readline = rl.readline("WOPR > ");
        match readline {
            Ok(line) => {
                if line.trim().is_empty() {
                    warn!(reason = "empty_prompt", "Ignoring empty prompt");
                } else {
                    turn += 1;
                    let input_chars = line.chars().count();
                    messages.push(ChatMessage::user().content(line).build());
                    // Send chat request and handle the response
                    chat_interaction(&*llm, &mut messages, turn, input_chars).await;
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

async fn chat_interaction(
    llm: &dyn LLMProvider,
    messages: &mut Vec<ChatMessage>,
    turn: u32,
    input_chars: usize,
) {
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

    let response_result = llm.chat(messages).instrument(turn_span.clone()).await;

    match response_result {
        Ok(response) => {
            // Print the response text
            if let Some(text) = response.text() {
                println!("Response: {text}");
                turn_span.record("response_chars", text.chars().count());
                turn_span.record("outcome", "ok");
                messages.push(ChatMessage::assistant().content(text).build());
            }
            // Print usage information
            if let Some(usage) = response.usage() {
                turn_span.record("prompt_tokens", usage.prompt_tokens);
                turn_span.record("completion_tokens", usage.completion_tokens);
                turn_span.in_scope(|| {
                    info!(
                        prompt_tokens = usage.prompt_tokens,
                        completion_tokens = usage.completion_tokens,
                        "Chat token usage"
                    );
                });
            } else {
                turn_span.in_scope(|| info!("Chat provider returned no usage information"));
            }
        }
        Err(err) => {
            turn_span.record("outcome", "provider_error");
            turn_span.in_scope(|| error!(error = %err, "Chat request failed"));
        }
    }
    turn_span.record("messages_after", messages.len());
    turn_span.in_scope(|| {
        debug!("Chat turn completed");
    });
}

#[cfg(not(feature = "open-ai-api"))]
fn build_llm() -> Box<dyn LLMProvider> {
    LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM provider
        .model("devstral_128k_ctx:latest")
        .max_tokens(512)
        .temperature(0.7)
        .normalize_response(true)
        .system("You are an expert software engineer that uses Rust as their main programming language.")
        .build()
        .expect("Failed to build LLM")
}

#[cfg(feature = "open-ai-api")]
fn build_llm() -> Box<dyn LLMProvider> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gpt-4.1-nano") // Use GPT-4.1 Nano model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .normalize_response(true) // Increase response normalization (e.g. function call stream)
        .system("You are an expert software engineer that uses Rust as their main programming language.")
        .build()
        .expect("Failed to build LLM")
}
