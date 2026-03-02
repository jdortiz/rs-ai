use std::{
    collections::HashMap,
    fs::{self, read_to_string},
    path::Path,
};

use anyhow::{Result, bail};
use log::{debug, error, info, warn};
use rig::{
    agent::Agent,
    client::CompletionClient,
    completion::{Completion, CompletionModel, Message},
    message::AssistantContent,
};
#[cfg(not(feature = "open-ai-api"))]
use rig::{client::Nothing, providers::ollama};
#[cfg(feature = "open-ai-api")]
use rig::{client::ProviderClient, providers::openai};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::Deserialize;
use serde_json::{Map, Value, json};

const AGENT_RULES: &str = r#"
You are an AI agent expert in software engineering, that can perform tasks by using available tools.  You use Rust as your main programming language

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
    env_logger::init();

    #[cfg(not(feature = "open-ai-api"))]
    info!("Using =Ollama=");
    #[cfg(not(feature = "open-ai-api"))]
    let agent = {
        let client: ollama::Client = ollama::Client::new(Nothing)?;
        client
            .agent("qwen3-coder:30b") // There are constants like `ollama::LLAMA3_2`, but a string can be used
            .preamble(AGENT_RULES)
            .build()
    };
    #[cfg(feature = "open-ai-api")]
    let agent = openai::Client::from_env()
        .agent(openai::GPT_4O)
        .preamble(AGENT_RULES)
        .build();

    // Prepare conversation history with example messages
    let mut chat_history = vec![
        Message::user("Hi, my name is Jorge."),
        Message::assistant("Hi Jorge! How can I help you?"),
    ];

    let mut rl = DefaultEditor::new()?;
    println!("Greetings Professor Falken!\n\nHow are you feeling today?");
    // Interaction user -- agent
    loop {
        let readline = rl.readline("WOPR > ");

        match readline {
            Ok(line) => {
                if !line.trim().is_empty() {
                    match perform_agent_loop(&agent, &line, &mut chat_history).await {
                        Ok(agent_response) => {
                            info!("Agent response:\n[[[--\n{agent_response}\n--]]]");
                            info!("Processing.");
                        }
                        Err(err) => {
                            error!("Agent error: {err:?}");
                        }
                    }
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

/// Makes the agent iterate to achieve the goal.
async fn perform_agent_loop<C: CompletionModel>(
    agent: &Agent<C>,
    user_input: &str,
    chat_history: &mut Vec<Message>,
) -> Result<String> {
    const MAX_ITERATIONS: u32 = 20;
    let mut iteration: u32 = 0;
    let mut finish = false;
    let mut agent_input = String::from(user_input);
    loop {
        let mut agent_response = String::new();
        debug!(
            "Iteration {iteration} Task: {user_input}\nInput: {agent_input}\nHistory: {chat_history:?}"
        );
        let completion_response = agent
            .completion(&agent_input, chat_history.to_owned())
            .await?
            .send()
            .await?;
        let AssistantContent::Text(response) = completion_response.choice.first() else {
            bail!("Non textual response:\n{:?}", completion_response.choice);
        };
        let response = response.text();
        debug!("AI:\n<<<--\n{response}\n-->>>");
        let next_agent_input: String = match parse_action(response) {
            Ok(action_request) => {
                let result = match action_request.tool_name.as_str() {
                    "list_files" => {
                        debug!("Requested 'list_files' action");
                        list_files()
                    }
                    "read_file" => {
                        debug!("Requested 'read_file' action");
                        if let Some(file_name) = action_request.args.get("file_name") {
                            read_file(file_name)
                        } else {
                            json!({"error": "Missing file_name argument."})
                        }
                    }
                    "terminate" => {
                        debug!("Requested 'terminate' action");
                        finish = true;
                        terminate("That's all folks!")
                    }
                    _ => {
                        debug!("Requested unexpected action");
                        json!({"error": "unknown tool"})
                    }
                };
                debug!("Tool:\n[[[--\n{result}\n--]]]");
                info!(
                    "Iteration {iteration} Task: {user_input}\nInput: {agent_input}\nResponse: {response}"
                );
                if finish {
                    agent_response = result.to_string();
                }
                iteration += 1;
                if iteration >= MAX_ITERATIONS {
                    bail!("Too many iterations");
                }
                result.to_string()
            }
            Err(err) => {
                warn!("No more actions: {err}");
                agent_response = response.to_string();
                finish = true;

                String::new()
            }
        };
        // Add new information to memory
        chat_history.push(Message::user(agent_input));
        chat_history.push(Message::assistant(response));
        agent_input = next_agent_input;
        if finish {
            return Ok(agent_response);
        }
    }
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

    match read_to_string(&filename) {
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
    Value::Object(Map::new())
}

fn parse_action(response: &str) -> Result<Action> {
    let Some(action) = extract_action_block(response) else {
        warn!("Action missing: {response}");
        bail!("Err: Found no action");
    };
    debug!("Action: {action}");
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
