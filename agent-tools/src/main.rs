#[cfg(not(feature = "open-ai-api"))]
use rig_core::providers::ollama;
#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai;
use rig_core::{DynModel, completion::CompletionRequest, message::Message, operation::Completion};
use rustyline::{DefaultEditor, error::ReadlineError};
use tracing::{Instrument, debug, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say "I don't know".
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().expect("Unable to load configuration from .env");
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let startup_span = info_span!("startup");
    #[cfg(not(feature = "open-ai-api"))]
    let gen_model: DynModel<Completion> = startup_span.in_scope(|| {
        info!(
            provider = "ollama",
            generation_model = "devstral_128k_ctx:latest",
            "Configuring agent provider"
        );
        ollama::Ollama::new()
            .native_completion("devstral_128k_ctx:latest")
            .into()
    });
    #[cfg(feature = "open-ai-api")]
    let gen_model: DynModel<Completion> = startup_span.in_scope(|| {
        info!(
            provider = "openai",
            generation_model = "gpt-4o-mini",
            "Configuring agent provider"
        );
        openai::OpenAI::from_env()
            .expect("Unable to create OpenAI client instance")
            .chat(openai::GPT_4O_MINI)
            .into()
    });
    drop(startup_span);

    let mut chat_history = vec![
        Message::user("Hi, my name is Jorge."),
        Message::assistant("Hi Jorge! How can I help you?"),
    ];

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
                        outcome = field::Empty,
                        response_chars = field::Empty,
                    );

                    let response_result = gen_model
                        .call(
                            CompletionRequest::new(&line)
                                .preamble(AGENT_RULES)
                                .messages(chat_history.iter().cloned()),
                        )
                        .instrument(turn_span.clone())
                        .await;
                    match response_result {
                        Ok(response) => {
                            turn_span.record("outcome", "ok");
                            turn_span.record("response_chars", response.text().chars().count());
                            turn_span.in_scope(|| debug!("Agent response received"));
                            println!("{}", response.text());
                            chat_history.push(Message::user(line));
                            chat_history.push(Message::assistant(response.text()));
                        }
                        Err(err) => {
                            turn_span.record("outcome", "agent_error");
                            turn_span.in_scope(|| {
                                error!(turn, error = %err, "Agent prompt failed");
                            });
                            return Err(err.into());
                        }
                    }
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

/*
// tools
fn list_files() -> Value {
    match fs::read_dir(Path::new(".")) {
        Ok(paths) => {
            let files = paths
                .flatten()
                .map(|entry| format!("{:?}", entry.path()))
                .collect::<Vec<String>>();
            json!({"tool_name": "list_files",
                   "args": {},
                   "result": files})
        }
        Err(err) => json!({"tool_name": "list_files",
                           "args": {},
                           "result": err.to_string() }),
    }
}

fn read_file<P: AsRef<Path>>(filename: P) -> Value {
    debug!("Reading file: {:?}", filename.as_ref());

    match fs::read_to_string(&filename) {
        Ok(content) => json!({"tool_name": "read_file",
                              "args": { "file_name": filename.as_ref().to_str().unwrap_or_default() },
                              "result": content}),
        Err(err) => json!({ "tool_name": "read_file",
                             "args": { "file_name": filename.as_ref().to_str().unwrap_or_default() },
                             "result": err.to_string() }),
    }
}

fn terminate(msg: &str) -> Value {
    debug!("{msg}. Hasta la vista!");
    Value::Object(serde_json::Map::new())
}

fn parse_action(response: &str) -> anyhow::Result<Action> {
    let Some(action) = extract_action_block(response) else {
        warn!("Action missing: {response}");
        bail!("Err: Found no action");
    };
    debug!(action, "Parsed action");
    let action_request: Action = serde_json::from_str(&action)?;

    Ok(action_request)
}

pub fn extract_action_block(markdown: &str) -> Option<String> {
    let mut code_lines = Vec::new();
    let mut in_code_block = false;
    let mut found = false;

    for line in markdown.lines() {
        if in_code_block {
            if line.trim().starts_with("```") {
                // End of code block
                in_code_block = false;
                if found {
                    // Return the joined content of the code block
                    let code_content = code_lines.join("\n");
                    return Some(code_content);
                }
                // Clear the code_lines for any subsequent code blocks
                code_lines.clear();
                continue;
            } else {
                // Collect the line into the code block
                code_lines.push(line.to_string());
            }
        } else if line.trim().starts_with("```") {
            // Start of code block
            let lang = line.trim_start_matches("```").trim();
            if lang == "action" {
                in_code_block = true;
                found = true;
                code_lines.clear();
            }
        }
    }
    None
}
*/
