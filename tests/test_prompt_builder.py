from AutoThemeGenerator.core.prompt_builder import create_prompt


def test_create_prompt_includes_optional_sections():
    prompt = create_prompt(
        context="Context text",
        research_questions="Research questions text",
        script="Script text",
        text_chunk="Transcript chunk",
        prompt_type="transcript",
    )

    assert "Context of the Study" in prompt
    assert "Research Questions" in prompt
    assert "Script" in prompt
    assert "Transcript segment" in prompt


def test_create_prompt_without_script_hides_section():
    prompt = create_prompt(
        context="Some context",
        research_questions="Some questions",
        script=None,
        text_chunk="Chunk",
        prompt_type="themes_same_id",
    )
    assert "**Script:**" not in prompt
    assert "Themes to be Synthesized" in prompt
