use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};
use log::{error, info};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    env_logger::init();

    #[cfg(not(feature = "open-ai-api"))]
    info!("Using =Ollama=");

    // Initialize and configure the LLM client
    let llm = build_llm();

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user().content("Hello. Call me Jorge.").build(),
        ChatMessage::assistant()
            .content("Hello, Jorge! How can I help you today.")
            .build(),
        ChatMessage::user()
            .content("Write a declarative macro to create a HashMap using any number of tuples.")
            .build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(response) => {
            // Print the response text
            if let Some(text) = response.text() {
                println!("Response: {text}");
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
