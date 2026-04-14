from euriai.langchain import create_chat_model


def get_chat_model(
    api_key: str,
    model_name: str = "gpt-4.1-nano",
    temperature: float = 0.7
):
    return create_chat_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )


def ask_chat_model(chat_model, prompt: str):
    try:
        response = chat_model.invoke(prompt)

        # ✅ Handle different response types safely
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    except Exception as e:
        return f"❌ Error from AI model: {str(e)}"
