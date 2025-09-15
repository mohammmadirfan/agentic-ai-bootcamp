# Agentic AI Bootcamp Hub

A smart AI assistant that automatically picks the right tool for your questions. Ask about anything - current events, math problems, or documents - and it figures out how to help you best.

## What It Does

- **Web Search**: Gets you the latest information from the internet
- **Math Solver**: Handles calculations and equations  
- **Document Q&A**: Answers questions about your uploaded files
- **General Chat**: Explains concepts and helps with brainstorming

The AI automatically decides which tool to use based on your question, so you don't have to think about it.

## Quick Setup

1. **Get the code**
   ```bash
   git clone https://github.com/mohammmadirfan/agentic-ai-bootcamp
   cd agentic-ai-bootcamp-hub
   ```

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys**
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_key_here
   SERPER_API_KEY=your_serper_key_here
   ```

4. **Run it**
   ```bash
   streamlit run app.py
   ```

## Example Questions

- "What's happening with AI today?"
- "Calculate 15% tip on $67.50"
- "Solve for x: 2x + 5 = 17"
- "What does my document say about sales targets?"
- "Explain how solar panels work"

## How It Works

When you ask a question, the AI looks at it and decides which tool will give you the best answer. It's built with LangGraph to make smart routing decisions automatically.

```
Your Question → AI Router → Best Tool → Your Answer
```

## What You Need

- Python 3.8 or higher
- API keys from [Groq](https://console.groq.com) and [Serper](https://serper.dev)
- That's it!

## Project Structure

```
├── agent/           # The AI brain and tools
├── evaluation/      # Testing and benchmarks  
├── data/           # Your documents and logs
├── app.py          # The web interface
└── requirements.txt # What to install
```

## Adding Your Own Tools

Want to add a new capability? Just:

1. Create a new tool in `agent/tools/`
2. Add it to the controller
3. Update the routing logic
4. Test it out

## Contributing

Found a bug or want to add a feature? Pull requests are welcome! Just make sure your code is clean and tested.

## Need Help?

- Check the troubleshooting section below
- Look at existing issues on GitHub
- Open a new issue if you're stuck

## Common Issues

**"API key not found"** - Make sure your `.env` file is in the right place with valid keys

**"Too many requests"** - You might be hitting rate limits, try again in a few minutes

**"Can't process document"** - Check that your file is a supported format and under 10MB

## License

MIT License - use it however you want!

---

Built with modern AI tools to make your life easier ✨