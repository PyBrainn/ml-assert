import requests

from ml_assert.core.base import AssertionResult


class SlackAlerter:
    """Slack integration for sending alerts."""

    def __init__(self, webhook_url: str):
        """Initialize the Slack alerter."""
        self.webhook_url = webhook_url

    def send_alert(self, result: AssertionResult) -> None:
        """
        Send an alert to Slack based on assertion result.

        Args:
            result: The assertion result to send as an alert.
        """
        message = {
            "text": f"Assertion {'PASSED' if result.success else 'FAILED'}: {result.message}",
            "attachments": [
                {
                    "color": "good" if result.success else "danger",
                    "fields": [
                        {
                            "title": "Timestamp",
                            "value": result.timestamp.isoformat(),
                            "short": True,
                        },
                        {
                            "title": "Metadata",
                            "value": str(result.metadata),
                            "short": True,
                        },
                    ],
                }
            ],
        }

        response = requests.post(
            self.webhook_url, json=message, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
