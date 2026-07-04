# ---------------------------------------------------------------------------
# Scrolling log + progress display
# ---------------------------------------------------------------------------
from collections import deque
import logging
import threading

from tqdm.asyncio import tqdm


class ScrollingLogDisplay:
    """
    Renders a fixed progress bar plus a scrolling window of the last N
    log lines beneath it, so multi-threaded workers don't stomp on each
    other's output or the progress bar.

    Implemented as one "real" tqdm progress bar plus N text-only tqdm
    bars pinned to fixed screen positions below it. Each new message
    pushes into a deque and the fixed positions are redrawn from it.
    """

    def __init__(self, total: int, max_lines: int = 20, desc: str = "Processing") -> None:
        self._lock = threading.Lock()
        self.max_lines = max_lines
        self.lines: deque = deque(maxlen=max_lines)

        self.bar = tqdm(total=total, position=0, desc=desc, leave=True)
        self._line_bars = [
            tqdm(total=0, position=i + 1, bar_format="{desc}", leave=False)
            for i in range(max_lines)
        ]

    def log(self, message: str) -> None:
        """
        Push a new log line into the scrolling window.

        :param message: Line to display
        """
        with self._lock:
            self.lines.append(message)
            for i, line_bar in enumerate(self._line_bars):
                text = self.lines[i] if i < len(self.lines) else ""
                line_bar.set_description_str(text)
                line_bar.refresh()

    def advance(self, n: int = 1) -> None:
        """
        Advance the main progress bar.

        :param n: Number of steps to advance
        """
        with self._lock:
            self.bar.update(n)

    def close(self) -> None:
        """Tear down all bars cleanly."""
        for line_bar in self._line_bars:
            line_bar.close()
        self.bar.close()



# ---------------------------------------------------------------------------
# Scrolling log + progress display
# ---------------------------------------------------------------------------


class ScrollingLogHandler(logging.Handler):
    """
    A :class:`logging.Handler` that forwards formatted log records into a
    :class:`ScrollingLogDisplay` instead of writing them straight to
    ``stderr``, so they don't corrupt the tqdm progress bar.

    :param display: The display instance to forward formatted records to.
    """

    def __init__(self, display: ScrollingLogDisplay) -> None:
        super().__init__()
        self.display = display

    def emit(self, record: logging.LogRecord) -> None:
        """
        Format and forward a single log record.

        :param record: The log record emitted by the logging module.
        """
        try:
            msg = self.format(record)
            self.display.log(msg)
        except Exception:  # pragma: no cover - logging must never crash the app
            self.handleError(record)