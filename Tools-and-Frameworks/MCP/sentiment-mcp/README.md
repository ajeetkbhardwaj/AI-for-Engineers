---
title: Mcp Sentiment
emoji: 🔥
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
license: mit
short_description: simple mcp based application that have access  to tool
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Deployed MCP Sentiment APP : https://huggingface.co/spaces/ajeet9843/mcp-sentiment
 
---

# 🚀 Report: Building & Deploying `mcp-sentiment` on Hugging Face

## 1. Project Overview

The **`mcp-sentiment` app** is a lightweight MCP-enabled application that performs **sentiment analysis** on text using `TextBlob`. It exposes a **Gradio interface** with an MCP server enabled, making it usable both as a web app and as an MCP tool for AI agents.

---

## 2. Project Files

You need only **two files**:

### `app.py`

```python
import json
import gradio as gr
from textblob import TextBlob

def sentiment_analysis(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    result = {
        "polarity": round(sentiment.polarity, 2),  # -1 (negative) to 1 (positive)
        "subjectivity": round(sentiment.subjectivity, 2),  # 0 (objective) to 1 (subjective)
        "assessment": "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
    }

    return json.dumps(result)

# Create the Gradio interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs=gr.Textbox(),
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using TextBlob"
)

# Launch the interface and MCP server
if __name__ == "__main__":
    demo.launch(mcp_server=True)
```

### `requirements.txt`

```
gradio[mcp]
textblob
```

---

## 3. Local Testing

Before deployment, test locally:

```bash
pip install -r requirements.txt
python app.py
```

*  It starts a **Gradio UI** in your browser.
* It also starts an **MCP server** automatically.

---

## 4. Deploying on Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Choose:

   * **SDK**: `Gradio`
   * **Hardware**: `CPU` (free tier is enough)
4. Upload your files:

   * `app.py`
   * `requirements.txt`
5. Hugging Face automatically builds the environment.

---

## 5. Running & Using the App

* Once deployed, your app will have a URL like:

  ```
  https://huggingface.co/spaces/ajeet9843/mcp-sentiment
  ```
* You can open it in a browser → enter text → get JSON sentiment results.
* You can also connect via MCP client (e.g., `langchain_mcp_adapters`) using the Space’s endpoint.

---

## 6. Example Output

Input:

```
I love working with MCP apps!
```

Output:

```json
{
  "polarity": 0.5,
  "subjectivity": 0.6,
  "assessment": "positive"
}
```

---

## 7. Benefits of Hugging Face Deployment

✅ Free hosting on Hugging Face Cloud
✅ Easy MCP integration for agents
✅ Interactive UI + MCP server in one deployment
✅ Auto-dependency management via `requirements.txt`

 