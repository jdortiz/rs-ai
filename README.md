# rs-ai: experiments with agentic AI in Rust

## This Repo

This repository contains the code used during my workshops on AI with Rust and it is based on my own talk for the Talent
Arena MWC about creating AI applications with Rust.

## The Workshop

During the workshop you build three small AI applications, each one a step closer to an agent:

- `chat-repl`: an interactive chat with conversation memory, using the `llm` crate.
- `rag`: Retrieval-Augmented Generation with an in-memory vector store, using `rig`.
- `agent-tools`: an agent that uses tools to work on your files, first with a hand-written agent loop and then with
  native tool calling.

The `main` branch contains the **starting point**. Follow the step-by-step guide in
**[INSTRUCTIONS.md](./INSTRUCTIONS.md)**, which also explains the prerequisites (Rust, Ollama or an OpenAI key) that you
should set up **before** the workshop.

The finished version of every stage lives in the `solution` branch, with one tagged commit per stage. To run it:

```shell
git switch solution
RUST_LOG=agent_tools=debug cargo r -p agent-tools
```

## License

This project is licensed under the terms of the [Apache license 2.0](./LICENSE.txt).

## Author

Jorge D. Ortiz Fuentes, 2026

## Resources

To learn more about Jorge's work you have all these fantastic resources:

- [🖋️ Jorge's Blog](https://jorgeortiz.dev/)
- [🙋🏻‍♂️ Jorge's LinkedIn](https://www.linkedin.com/in/jorgeortiz/)
- [🙋🏻‍♂️ Jorge's Reddit](https://www.reddit.com/user/jorgedortiz/)
- [🙋🏻‍♂️ Jorge's Bluesky](https://bsky.app/profile/jdortiz.bsky.social)
- [🙋🏻‍♂️ Jorge's Mastodon](https://fosstodon.org/@jdortiz)
- [🙋🏻‍♂️ Jorge's X](https://x.com/jdortiz)
