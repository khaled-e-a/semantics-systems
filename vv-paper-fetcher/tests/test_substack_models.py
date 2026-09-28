from pathlib import Path

from src.substack_post.models import (
    DigestEntry,
    DraftResult,
    Figure,
    PaperDetails,
    PostDraft,
    UploadedImage,
    Verdict,
    VerifyResult,
)


def test_verify_result_ok_reflects_errors():
    assert VerifyResult().ok is True
    assert VerifyResult(errors=["x"]).ok is False
    assert VerifyResult(warnings=["only a warning"]).ok is True


def test_all_dataclasses_construct():
    entry = DigestEntry(rank=1, section="top", title="T", url="https://arxiv.org/abs/2609.28614v1")
    assert entry.arxiv_id is None and entry.tags == []

    details = PaperDetails(
        arxiv_id="2609.28614", version="v2", title="T", abstract="A", authors=["Alice"]
    )
    assert details.license_url is None

    verdict = Verdict(url=entry.url, keep=False, rationale="off-topic")
    assert verdict.summary == "" and verdict.contribution_type == ""

    figure = Figure(
        url=entry.url,
        path=Path("fig.png"),
        figure_number="1",
        credit_caption="Figure 1 from “T” (arXiv:2609.28614)",
        source="html",
        openly_licensed=True,
        width=800,
        height=600,
        mime="image/png",
    )
    assert figure.width == 800

    draft = PostDraft(title="Title", intro="Intro", bodies={entry.url: "Body."})
    assert draft.bodies[entry.url] == "Body."

    image = UploadedImage(
        url="https://substackcdn.com/x.png", width=800, height=600, bytes=1234, content_type="image/png"
    )
    assert image.bytes == 1234

    result = DraftResult(
        draft_id=42, edit_url="https://example.substack.com/publish/post/42", created=True
    )
    assert result.created is True
