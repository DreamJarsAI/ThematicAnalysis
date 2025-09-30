from pathlib import Path

from AutoThemeGenerator.core.agent_pipeline import ThematicAnalysisResult
from AutoThemeGenerator.core.storage import write_analysis_outputs


def test_write_analysis_outputs_creates_expected_downloads(tmp_path: Path) -> None:
    result = ThematicAnalysisResult(
        chunk_texts=[[" First chunk  ", "Second chunk"]],
        chunk_outputs=[[" Theme 1  ", "Theme 2"]],
        participant_summaries=[[" Participant summary "]],
        overall_summaries=[" Overall theme "],
    )

    record = write_analysis_outputs(
        result=result,
        participant_names=["InterviewOne"],
        output_dir=tmp_path,
    )

    expected_files = {
        "InterviewOne_chunks.txt": "Chunk 1\nFirst chunk\n\nChunk 2\nSecond chunk",
        "InterviewOne_themes.txt": (
            "Chunk-specific themes\n\nChunk 1\nTheme 1\n\nChunk 2\nTheme 2\n\n"
            "Participant themes\n\nTheme block 1\nParticipant summary"
        ),
        "study_level_themes.txt": "Study-level themes\n\nTheme block 1\nOverall theme",
    }

    assert [file.name for file in record.files] == list(expected_files.keys())

    for file in record.files:
        assert file.path.read_text(encoding="utf-8") == expected_files[file.name]

    archive_names = {file.name for file in record.files}
    assert archive_names == set(expected_files)
