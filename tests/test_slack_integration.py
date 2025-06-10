from datetime import datetime
from unittest.mock import patch

from ml_assert.core.base import AssertionResult
from ml_assert.integrations.slack import SlackAlerter


def test_slack_alerter_send_alert():
    """Test that the SlackAlerter sends an alert correctly."""
    with patch("requests.post") as mock_post:
        alerter = SlackAlerter(webhook_url="http://example.com/webhook")
        result = AssertionResult(
            success=False,
            message="Test alert",
            timestamp=datetime.now(),
            metadata={"foo": "bar"},
        )
        alerter.send_alert(result)
        mock_post.assert_called_once()
