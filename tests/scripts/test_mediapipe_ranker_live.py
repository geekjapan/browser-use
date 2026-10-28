"""On-device MediaPipe ranker check; needs the 165 MB EmbeddingGemma 2 model, so it is not in tests/ci.

MEDIAPIPE_DECISION_MODEL_PATH=embeddinggemma-2-text-270m.litertlm \
	uv run --extra mediapipe pytest tests/scripts/test_mediapipe_ranker_live.py
"""

import os

import pytest


async def test_mediapipe_ranker_live():
	"""Runs only with MEDIAPIPE_DECISION_MODEL_PATH, e.g. embeddinggemma-2-text-270m.litertlm."""
	path = os.getenv('MEDIAPIPE_DECISION_MODEL_PATH')
	if not path:
		pytest.skip('Set MEDIAPIPE_DECISION_MODEL_PATH to run the on-device ranker.')
	pytest.importorskip('mediapipe.tasks.python.decision.decision_maker')
	from browser_use.decision import MediaPipeTargetRanker

	ranker = MediaPipeTargetRanker(path)
	try:
		scores = await ranker.rank(
			'Open the privacy policy',
			{
				'click:1': '{"tag": "a", "text": "Home"}',
				'click:2': '{"tag": "a", "text": "Privacy policy"}',
				'click:3': '{"tag": "button", "text": "Add to cart: Blue Widget"}',
			},
		)
	finally:
		ranker.close()
	assert max(scores, key=scores.__getitem__) == 'click:2'
