use std::{fs, io, path::Path};

use rig_agent::tool::{Tool, ToolContext};
use rig_agent::{Agent, AgentBuilder};
#[cfg(not(feature = "open-ai-api"))]
use rig_core::providers::ollama;
#[cfg(feature = "open-ai-api")]
use rig_core::providers::openai;
use rig_core::{DynModel, message::Message, operation::Completion};
use rustyline::{DefaultEditor, error::ReadlineError};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{Instrument, Span, debug, error, field, info, info_span, warn};
use tracing_subscriber::EnvFilter;

const AGENT_RULES: &str = r#"
You are an AI agent expert in software engineering, and you use Rust as your main programming language.

Use the provided tools when the user asks about files. Use list_files before read_file when you need to find a file.
When you have enough information, answer the user directly without calling another tool.
"#;

#[derive(Deserialize)]
struct NoArguments {}

struct ListFiles;

impl Tool for ListFiles {
    const NAME: &'static str = "list_files";
    type Args = NoArguments;
    type Output = Vec<String>;
    type Error = io::Error;

    fn description(&self) -> String {
        "List all files in the current directory.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: NoArguments,
    ) -> Result<Self::Output, Self::Error> {
        list_files()
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {
    file_name: String,
}

struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Args = ReadFileArgs;
    type Output = String;
    type Error = io::Error;

    fn description(&self) -> String {
        "Read the content of a file by its path.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The path of the file to read."
                }
            },
            "required": ["file_name"],
            "additionalProperties": false
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: ReadFileArgs,
    ) -> Result<Self::Output, Self::Error> {
        read_file(args.file_name)
    }
}

#[derive(Deserialize)]
struct TerminateArgs {
    message: String,
}

struct Terminate;

impl Tool for Terminate {
    const NAME: &'static str = "terminate";
    type Args = TerminateArgs;
    type Output = String;
    type Error = io::Error;

    fn description(&self) -> String {
        "End the task and provide a final summary message.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The final summary to show the user."
                }
            },
            "required": ["message"],
            "additionalProperties": false
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: TerminateArgs,
    ) -> Result<Self::Output, Self::Error> {
        Ok(terminate(&args.message))
    }
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
    let agent = startup_span.in_scope(|| {
        AgentBuilder::new(gen_model)
            .preamble(AGENT_RULES)
            .default_max_turns(20)
            .tool(ListFiles)
            .tool(ReadFile)
            .tool(Terminate)
            .build()
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
                    match perform_agent_loop(&agent, &line, &mut chat_history, &turn_span).await {
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
                warn!(reason = "interrupt", "Ctrl-C interrupted");
                break;
            }
            Err(ReadlineError::Eof) => {
                warn!(reason = "eof", "Ctrl-D finished");
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

async fn perform_agent_loop(
    agent: &Agent,
    user_input: &str,
    chat_history: &mut Vec<Message>,
    turn_span: &Span,
) -> anyhow::Result<String> {
    turn_span.in_scope(|| info!("Agent processing user request."));

    let response = agent
        .chat(user_input.to_owned(), chat_history)
        .instrument(turn_span.clone())
        .await?;

    Ok(response.output())
}

fn list_files() -> io::Result<Vec<String>> {
    fs::read_dir(Path::new("."))?
        .map(|entry| entry.map(|entry| format!("{:?}", entry.path())))
        .collect()
}

fn read_file<P: AsRef<Path>>(filename: P) -> io::Result<String> {
    debug!(path = ?filename.as_ref(), "Reading file");
    fs::read_to_string(filename)
}

fn terminate(msg: &str) -> String {
    debug!(message = msg, "Terminating agent task");
    msg.to_owned()
}
