"""
Debug utilities for tmcmc package.

This package contains debugging and logging functionality extracted from
case2_tmcmc_linearization.py:
- logger: DebugLogger class with hook-based control and Slack notifications
"""

from .logger import (
    DebugLogger,
    SLACK_ENABLED,
    notify_slack,
    SlackNotifier,
    _slack_notifier,
)

__all__ = [
    "DebugLogger",
    "SLACK_ENABLED",
    "notify_slack",
    "SlackNotifier",
    "_slack_notifier",
]
