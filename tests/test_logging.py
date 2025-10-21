"""
Test logging

Test the logging features.
"""

import logging

from frg.utils.utils import get_logger


def test_logging(caplog):
    logger = get_logger(__name__)
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    assert logger.level == logging.DEBUG

    assert len(caplog.records) == 5
    assert caplog.records[0].levelname == "DEBUG"
    assert caplog.records[1].levelname == "INFO"
    assert caplog.records[2].levelname == "WARNING"
    assert caplog.records[3].levelname == "ERROR"
    assert caplog.records[4].levelname == "CRITICAL"

    assert caplog.records[0].message == "This is a DEBUG message"
    assert caplog.records[1].message == "This is an INFO message"
    assert caplog.records[2].message == "This is a WARNING message"
    assert caplog.records[3].message == "This is an ERROR message"
    assert caplog.records[4].message == "This is a CRITICAL message"
