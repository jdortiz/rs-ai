use llm::{
    LLMProvider,
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};
use tracing::{error, info, info_span};
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
    let messages = vec![
        ChatMessage::user().content("Hello. Call me Jorge.").build(),
        ChatMessage::assistant()
            .content("Hello, Jorge! How can I help you today?")
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
                info!(
                    prompt_tokens = usage.prompt_tokens,
                    completion_tokens = usage.completion_tokens,
                    "Chat token usage"
                );
            } else {
                info!("Chat provider returned no usage information");
            }
        }
        Err(err) => error!(error = %err, "Chat request failed"),
    }
    Ok(())
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
