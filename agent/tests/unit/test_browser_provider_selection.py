"""Which cloud-browser vendor a deployment ends up on.

Browserbase became the default, and most existing deployments have only a
Browser Use key. Getting this wrong is not a startup error the operator sees --
it is the browser panel quietly not working, or worse, working on a vendor
nobody chose, where the user is signed out of every site they had signed in to.
"""

from __future__ import annotations

import pytest

from src.browser.providers import (
    BROWSER_USE,
    BROWSERBASE,
    LaunchedBrowser,
    ProviderUnavailableError,
    configured_vendors,
    create_browser_provider,
)
from src.settings import Settings


def _settings(**kwargs) -> Settings:
    return Settings(**kwargs)


def test_an_existing_browser_use_deployment_keeps_working() -> None:
    """The default changed to Browserbase; deployments did not change with it."""
    provider = create_browser_provider(_settings(browser_use_api_key="bu_key"))
    assert provider.vendor == BROWSER_USE


def test_browserbase_is_used_when_it_is_configured() -> None:
    provider = create_browser_provider(
        _settings(browserbase_api_key="bb_key", browserbase_project_id="p1")
    )
    assert provider.vendor == BROWSERBASE


def test_the_requested_vendor_wins_when_both_are_configured() -> None:
    settings = _settings(
        browser_provider=BROWSER_USE,
        browserbase_api_key="bb_key",
        browser_use_api_key="bu_key",
    )
    assert create_browser_provider(settings).vendor == BROWSER_USE


def test_the_default_wins_when_both_are_configured_and_none_is_requested() -> None:
    settings = _settings(browserbase_api_key="bb_key", browser_use_api_key="bu_key")
    assert create_browser_provider(settings).vendor == BROWSERBASE


def test_an_unknown_vendor_name_falls_back_rather_than_failing(caplog) -> None:
    settings = _settings(browser_provider="playwright", browser_use_api_key="bu_key")
    assert create_browser_provider(settings).vendor == BROWSER_USE


def test_no_credentials_at_all_is_an_error_naming_both_options() -> None:
    with pytest.raises(ProviderUnavailableError) as excinfo:
        create_browser_provider(_settings())
    message = str(excinfo.value)
    assert "BROWSERBASE_API_KEY" in message
    assert "BROWSER_USE_API_KEY" in message


def test_falling_back_says_so(caplog) -> None:
    """Silently running on the other vendor logs the user out of everything."""
    import logging

    with caplog.at_level(logging.WARNING):
        create_browser_provider(
            _settings(browser_provider=BROWSERBASE, browser_use_api_key="bu_key")
        )

    assert any("falling back" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (Settings(), []),
        (Settings(browserbase_api_key="k"), [BROWSERBASE]),
        (Settings(browser_use_api_key="k"), [BROWSER_USE]),
        (
            Settings(browserbase_api_key="k", browser_use_api_key="k"),
            [BROWSERBASE, BROWSER_USE],
        ),
    ],
)
def test_configured_vendors_reports_what_can_actually_run(settings, expected) -> None:
    assert configured_vendors(settings) == expected


def test_a_launched_browser_must_be_reachable() -> None:
    """A browser with no CDP endpoint is a billed resource nothing can drive."""
    with pytest.raises(ValueError, match="CDP endpoint"):
        LaunchedBrowser(browser_id="b1", vendor=BROWSERBASE, live_url="https://live")
