use std::{collections::HashMap, fs, path::Path};

use anyhow::{anyhow, bail};
#[cfg(not(feature = "open-ai-api"))]
use rig_core::providers::ollama;
#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai;
use rig_core::{DynModel, completion::CompletionRequest, message::Message, operation::Completion};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{Instrument, Span, debug, debug_span, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;

const AGENT_RULES: &str = r#"
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
"#;

#[derive(Deserialize)]
struct Action {
    tool_name: String,
    args: HashMap<String, String>,
}

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

/// Makes the agent iterate to achieve the goal.
async fn perform_agent_loop(
    gen_model: &DynModel<Completion>,
    user_input: &str,
    chat_history: &mut Vec<Message>,
    turn_span: &Span,
) -> anyhow::Result<String> {
    turn_span.in_scope(|| {
        info!("Agent processing user request.");
    });

    const MAX_ITERATIONS: u32 = 20;
    let mut agent_input = String::from(user_input);

    for iteration in 1..=MAX_ITERATIONS {
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

        let response_result = gen_model
            .call(
                CompletionRequest::new(&agent_input)
                    .preamble(AGENT_RULES)
                    .messages(chat_history.iter().cloned()),
            )
            .instrument(iteration_span.clone())
            .await;

        match response_result {
            Ok(response) => {
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

                let is_termination = action_request.tool_name == "terminate";
                let result = iteration_span.in_scope(|| match action_request.tool_name.as_str() {
                    "list_files" => {
                        debug!(action = "list_files", "Executing action");
                        list_files()
                    }
                    "read_file" => {
                        debug!(action = "read_file", "Executing action");
                        if let Some(file_name) = action_request.args.get("file_name") {
                            read_file(file_name)
                        } else {
                            json!({"error": "Missing file_name argument."})
                        }
                    }
                    "terminate" => {
                        debug!(action = "terminate", "Executing action");
                        terminate(
                            action_request
                                .args
                                .get("message")
                                .map(String::as_str)
                                .unwrap_or("That's all folks!"),
                        )
                    }
                    _ => {
                        debug!("Requested unexpected action");
                        json!({"error": "unknown tool"})
                    }
                });

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
    }

    turn_span.record("outcome", "iteration_limit");
    Err(anyhow!(
        "Agent exceeded the maximum of {MAX_ITERATIONS} iterations"
    ))
}

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
