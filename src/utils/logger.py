"""Ring buffer logging system with line and file limits."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from collections import deque
import os


class RingBufferHandler(logging.Handler):
    """
    Custom logging handler that implements ring buffer functionality.

    Maintains a fixed number of log files, each with a maximum number of lines.
    When limits are exceeded, oldest logs are discarded.
    """

    def __init__(
        self,
        log_dir: Path,
        log_name: str = "app.log",
        max_lines: int = 10000,
        max_files: int = 5,
        level: int = logging.INFO
    ):
        """
        Initialize ring buffer handler.

        Args:
            log_dir: Directory to store log files
            log_name: Base name for log files
            max_lines: Maximum lines per log file
            max_files: Maximum number of log files to keep
            level: Logging level
        """
        super().__init__(level)
        self.log_dir = Path(log_dir)
        self.log_name = log_name
        self.max_lines = max_lines
        self.max_files = max_files

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current log file and line count
        self.current_file_index = 0
        self.current_line_count = 0

        # Initialize first log file
        self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """Initialize or switch to a new log file."""
        self.current_log_path = self.log_dir / f"{self.log_name}.{self.current_file_index}"
        self.current_line_count = 0

        # Clean up old files if exceeding max_files
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove oldest log files if exceeding max_files limit."""
        log_files = sorted(self.log_dir.glob(f"{self.log_name}.*"))

        if len(log_files) >= self.max_files:
            # Remove oldest files
            files_to_remove = log_files[: len(log_files) - self.max_files + 1]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except OSError:
                    pass

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the ring buffer.

        Args:
            record: Log record to emit
        """
        try:
            # Check if we need to rotate to a new file
            if self.current_line_count >= self.max_lines:
                self.current_file_index += 1
                self._initialize_log_file()

            # Format and write the log message
            msg = self.format(record)

            with open(self.current_log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

            self.current_line_count += 1

        except Exception:
            self.handleError(record)


def setup_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
    max_lines: int = 10000,
    max_files: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with ring buffer handler.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        max_lines: Maximum lines per log file
        max_files: Maximum number of log files
        console_output: Whether to also output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers

    # Ring buffer file handler
    file_handler = RingBufferHandler(
        log_dir=log_dir,
        log_name=name,
        max_lines=max_lines,
        max_files=max_files,
        level=level
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
