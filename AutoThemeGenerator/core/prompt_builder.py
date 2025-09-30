from __future__ import annotations

from nltk.tokenize import word_tokenize


def create_prompt(
    context: str,
    research_questions: str,
    script: str | None,
    text_chunk: str,
    prompt_type: str,
) -> str:
    if prompt_type == "transcript":
        objective = (
            "Given the extensive length of the transcripts, they have been segmented to "
            "facilitate a more manageable analysis. Your task involves three objectives, each "
            "focused on analyzing and representing the themes from the transcript segment:\n"
            "1. Identify Main Themes: Begin by discerning the main themes. For each one, "
            "formulate a concise topic sentence that encapsulates its essence.\n"
            "2. Explain the Themes: Subsequently, for each identified theme, provide a detailed "
            "explanation that elaborates on its significance and context.\n"
            "3. Illustrative Quotes: Select one impactful quote from the transcript segment that "
            "accurately exemplifies each theme. Ensure that the chosen quote is a representative "
            "embodiment of the theme it is paired with.\n"
        )
        action = "**Transcript segment:**\n" + text_chunk
        quote = "illustrative quote"
    elif prompt_type == "themes_same_id":
        objective = (
            "Previously, you identified, elaborated, and exemplified various themes from each "
            "transcript segment. Each theme was meticulously detailed, involving a concise topic "
            "sentence, a comprehensive explanation, and a selected illustrative quote.\nNow, a more "
            "synthesized analysis is required. Having compiled the segmented analyses into a unified "
            "document, your task is to critically examine the identified themes across the segments. "
            "Assess the themes for similarities and differences, aiming to integrate and synthesize "
            "them into a more concise set of distinct themes.\nPlease refine and consolidate the "
            "themes, reducing redundancy and emphasizing uniqueness and significance.\n"
        )
        action = "**Themes to be Synthesized:**\n" + text_chunk
        quote = "illustrative quote"
    else:
        objective = (
            "You have previously summarized various themes from each study participant's "
            "transcript. Your next task is to meticulously evaluate these themes, identifying "
            "similarities and differences across participants. Your aim should be to integrate and "
            "condense them into a more concise and distinct set of themes.\nPay special attention to "
            "identifying and synthesizing the most prevalent themes among participants. Present the "
            "identified themes in the descending order of popularity, showcasing the most common "
            "themes first, followed by the less common ones.\nMake sure each theme is relevant to the "
            "study's research question."
        )
        action = "**Themes to be Synthesized:**\n" + text_chunk
        quote = "2-3 illustrative quotes"

    if script is None:
        prompt_template = (
            "I am a university professor, seeking assistance from a research assistant. Our "
            "research team has conducted several interviews and focus group sessions, utilizing a "
            "semi-structured script, and the audio from these sessions has been transcribed into text.\n"
            "Here are the essential background details of the study:\n"
            "**Context of the Study:**\n"
            f"{context}\n"
            "**Research Questions:**\n"
            f"{research_questions}\n"
            "**Objective:**\n"
            f"{objective}"
            f"{action}\n"
            "**Format of Your Response:**\n"
            "Structure your response by delineating each theme separately and sequentially. For each "
            "theme, follow this format:\n"
            "Theme 1:\n"
            "- Topic Sentence: [Your succinct topic sentence here.]\n"
            "- Explanation: [Your comprehensive explanation here.]\n"
            f"- Quote: '[Your chosen {quote} here.]'\n"
            "Theme 2:\n"
            "- Topic Sentence: ...\n"
            "Continue in the same manner for each subsequent theme, organizing the information "
            "clearly and coherently. Please exclude unnecessary information, such as descriptions "
            "preceding each theme (e.g., Theme 1)."
            "Please aim to provide an optimal number of themes, make them as concise as possible and "
            "ensure their relevance to the study's research question."
        )
    else:
        prompt_template = (
            "I am a university professor, seeking assistance from a research assistant. Our "
            "research team has conducted several interviews and focus group sessions, utilizing a "
            "semi-structured script, and the audio from these sessions has been transcribed into text.\n"
            "Here are the essential background details of the study:\n"
            "**Context of the Study:**\n"
            f"{context}\n"
            "**Research Questions:**\n"
            f"{research_questions}\n"
            "**Script:**\n"
            f"{script}\n"
            "**Objective:**\n"
            f"{objective}"
            f"{action}\n"
            "**Format of Your Response:**\n"
            "Structure your response by delineating each theme separately and sequentially. For each "
            "theme, follow this format:\n"
            "Theme 1:\n"
            "- Topic Sentence: [Your succinct topic sentence here.]\n"
            "- Explanation: [Your comprehensive explanation here.]\n"
            f"- Quote: '[Your chosen {quote} here.]'\n"
            "Theme 2:\n"
            "- Topic Sentence: ...\n"
            "Continue in the same manner for each subsequent theme, organizing the information "
            "clearly and coherently. Please exclude unnecessary information, such as descriptions "
            "preceding each theme (e.g., Theme 1)."
            "Please aim to provide an optimal number of themes, make them as concise as possible and "
            "ensure their relevance to the study's research question."
        )

    return prompt_template


def count_prompt_tokens(*segments: str) -> int:
    return sum(len(_safe_tokenize(segment)) for segment in segments if segment)


def _safe_tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


__all__ = ["create_prompt", "count_prompt_tokens"]
