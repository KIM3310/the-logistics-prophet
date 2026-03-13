from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "site" / "index.html"
PREVIEW_CARD = ROOT / "site" / "preview-card.svg"


class FrontendMetadataTests(unittest.TestCase):
    def test_frontend_metadata_contract(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")
        required_tokens = [
            'name="description"',
            'property="og:title"',
            'property="og:description"',
            'property="og:image"',
            'property="og:image:alt"',
            'name="twitter:title"',
            'name="twitter:description"',
            'name="twitter:image"',
            'name="google-adsense-account"',
        ]

        for token in required_tokens:
            self.assertIn(token, html, token)

    def test_frontend_reviewer_path_contract(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")
        required_tokens = [
            "Reviewer continuity, not runtime theater",
            "id=\"laneFocusPanel\"",
            "id=\"copyLaneFocusBtn\"",
            "Hold one risky lane from baseline through handoff.",
            "Public site = reviewer map",
            "recovery drill summary",
            "owner / ETA workflow",
            "This public surface summarizes the proof route",
        ]

        for token in required_tokens:
            self.assertIn(token, html, token)

    def test_preview_asset_exists(self) -> None:
        self.assertTrue(PREVIEW_CARD.exists())


if __name__ == "__main__":
    unittest.main()
