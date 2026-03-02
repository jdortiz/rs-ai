use std::{
    fs::{self, read_to_string},
    path::Path,
};

use anyhow::{Result, bail};
use log::{debug, error, info, warn};
use rig::{
    client::CompletionClient,
    completion::{Completion, CompletionModel, Message},
    message::AssistantContent,
};
#[cfg(not(feature = "open-ai-api"))]
use rig::{client::Nothing, providers::ollama};
#[cfg(feature = "open-ai-api")]
use rig::{client::ProviderClient, providers::openai};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde_json::{Map, Value, json};

const AGENT_RULES: &str = r#"
You are an expert software engineer that uses Rust as their main programming language.

If you don't know the answer, say \"I don't know\".
"#;

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
                    let completion_response = agent
                        .completion(&line, chat_history.to_owned())
                        .await?
                        .send()
                        .await?;
                    let AssistantContent::Text(response) = completion_response.choice.first()
                    else {
                        bail!("Non textual response:\n{:?}", completion_response.choice);
                    };
                    let response = response.text();

                    println!("{response}");
                    // Add new information to memory
                    chat_history.push(Message::user(line));
                    chat_history.push(Message::assistant(response));
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
*/
