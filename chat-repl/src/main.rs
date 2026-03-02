use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};
use log::{error, info, warn};
use rustyline::{DefaultEditor, error::ReadlineError};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    env_logger::init();

    #[cfg(not(feature = "open-ai-api"))]
    info!("Using =Ollama=");

    // Initialize and configure the LLM client
    let llm = build_llm();

    // Prepare conversation history with example messages
    let mut messages = vec![
        ChatMessage::user().content("Hello. Call me Jorge.").build(),
        ChatMessage::assistant()
            .content("Hello, Jorge! How can I help you today.")
            .build(),
    ];

    let mut rl = DefaultEditor::new()?;
    println!("Greetings Professor Falken!\n\nHow are you feeling today?");
    loop {
        let readline = rl.readline("WOPR > ");
        match readline {
            Ok(line) => {
                if !line.trim().is_empty() {
                    messages.push(ChatMessage::user().content(line).build());
                    chat_interaction(&llm, &mut messages).await;
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

#[cfg(not(feature = "open-ai-api"))]
fn build_llm() -> Box<dyn LLMProvider> {
    LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM provider
        .model("qwen3-coder:30b")
        .max_tokens(512)
        .temperature(0.7)
        .normalize_response(true)
        .system("You are an expert software engineer that uses Rust as their main programming language.")
        .build()
        .expect("Failed to build LLM")
}

async fn chat_interaction(llm: &Box<dyn LLMProvider>, messages: &mut Vec<ChatMessage>) {
    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(response) => {
            // Print the response text
            if let Some(text) = response.text() {
                println!("Response: {text}");
                messages.push(ChatMessage::assistant().content(text).build());
            }
            // Print usage information
            if let Some(usage) = response.usage() {
                info!("  Prompt tokens: {}", usage.prompt_tokens);
                info!("  Completion tokens: {}", usage.completion_tokens);
            } else {
                info!("No usage information available");
            }
        }
        Err(e) => error!("Chat error: {e}"),
    }
}
