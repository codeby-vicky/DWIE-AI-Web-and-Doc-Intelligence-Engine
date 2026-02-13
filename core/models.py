from langchain_community.llms import Ollama


def get_llm(mode):

    if mode == "Research Mode":
        return Ollama(
            model="mistral",
            temperature=0.2,
            num_predict=800
        )

    elif mode == "Study Mode":
        return Ollama(
            model="qwen:4b",
            temperature=0.4,
            num_predict=500
        )

    else:
        return Ollama(model="mistral")
