# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
Encoder Gateway — ZMQ proxy distributing requests from PD servers to encoder
servers using pluggable scheduling (round-robin, least-outstanding, data-size-balanced).

Architecture:
    PD0 (REQ) ─┐
    PD1 (REQ) ─┼──→ Gateway (ROUTER ←→ DEALER×N) ──scheduler──→ Encoder0 (ROUTER)
    PD2 (REQ) ─┘                                               → Encoder1 (ROUTER)

Dependencies: zmq, stdlib only.
"""

import abc
import argparse
import collections
import logging
import signal
import time

import zmq

from ksana_encoder.zmq_zero_copy import peek_stream_metadata
from utilize.logger import get_logger

logger = get_logger("ksana_encoder")


# Scheduler strategies


class SchedulerStrategy(abc.ABC):
    """Abstract base class for encoder scheduling strategies."""

    def __init__(self, num_encoders):
        self._num_encoders = num_encoders

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @abc.abstractmethod
    def select(self, data_frames) -> int:
        """Choose encoder index for the next request. Returns index in [0, num_encoders)."""

    @abc.abstractmethod
    def on_request_sent(self, encoder_idx, data_frames):
        """Called after a request has been forwarded to *encoder_idx*."""

    @abc.abstractmethod
    def on_response_received(self, encoder_idx):
        """Called after a response has been received from *encoder_idx*."""

    def stats_extra(self) -> str:
        """Optional extra info to append to periodic stats line."""
        return ""


class RoundRobinScheduler(SchedulerStrategy):
    """Simple round-robin: cycle through encoders in order."""

    def __init__(self, num_encoders):
        super().__init__(num_encoders)
        self._counter = 0

    @property
    def name(self) -> str:
        return "round-robin"

    def select(self, data_frames) -> int:
        chosen = self._counter % self._num_encoders
        self._counter += 1
        return chosen

    def on_request_sent(self, encoder_idx, data_frames):
        pass

    def on_response_received(self, encoder_idx):
        pass


class LeastOutstandingScheduler(SchedulerStrategy):
    """Least-outstanding: pick the encoder with fewest in-flight requests; ties broken by round-robin."""

    def __init__(self, num_encoders):
        super().__init__(num_encoders)
        self._inflight = [0] * num_encoders
        self._rr_tiebreak = 0

    @property
    def name(self) -> str:
        return "least-outstanding"

    def select(self, data_frames) -> int:
        min_inflight = min(self._inflight)
        candidates = [
            i for i in range(self._num_encoders) if self._inflight[i] == min_inflight
        ]
        chosen = candidates[self._rr_tiebreak % len(candidates)]
        self._rr_tiebreak += 1
        return chosen

    def on_request_sent(self, encoder_idx, data_frames):
        self._inflight[encoder_idx] += 1

    def on_response_received(self, encoder_idx):
        self._inflight[encoder_idx] -= 1

    def stats_extra(self) -> str:
        return "inflight=" + str(self._inflight)


class DataSizeBalancedScheduler(SchedulerStrategy):
    """Data-size-balanced: pick the encoder with least pending bytes (proxy for compute cost).

    Uses per-encoder FIFO deques to track request sizes for accurate subtraction on response;
    relies on encoders processing in strict FIFO order. Ties broken by round-robin.
    """

    def __init__(self, num_encoders):
        super().__init__(num_encoders)
        self._pending_bytes = [0] * num_encoders
        self._request_sizes = [collections.deque() for _ in range(num_encoders)]
        self._rr_tiebreak = 0

    @property
    def name(self) -> str:
        return "data-size-balanced"

    @staticmethod
    def _compute_data_size(data_frames) -> int:
        """Sum of all data frame byte lengths."""
        return sum(len(f) for f in data_frames)

    def select(self, data_frames) -> int:
        request_size = self._compute_data_size(data_frames)
        min_pending = min(
            self._pending_bytes[i] + request_size for i in range(self._num_encoders)
        )
        candidates = [
            i
            for i in range(self._num_encoders)
            if self._pending_bytes[i] + request_size == min_pending
        ]
        chosen = candidates[self._rr_tiebreak % len(candidates)]
        self._rr_tiebreak += 1
        return chosen

    def on_request_sent(self, encoder_idx, data_frames):
        size = self._compute_data_size(data_frames)
        self._pending_bytes[encoder_idx] += size
        self._request_sizes[encoder_idx].append(size)

    def on_response_received(self, encoder_idx):
        if not self._request_sizes[encoder_idx]:
            return
        size = self._request_sizes[encoder_idx].popleft()
        self._pending_bytes[encoder_idx] -= size

    def stats_extra(self) -> str:
        pending_mb = [f"{b / (1024 * 1024):.1f}MB" for b in self._pending_bytes]
        return "pending_bytes=" + str(pending_mb)


SCHEDULER_REGISTRY = {
    "round-robin": RoundRobinScheduler,
    "least-outstanding": LeastOutstandingScheduler,
    "data-size-balanced": DataSizeBalancedScheduler,
}
DEFAULT_SCHEDULER = "data-size-balanced"


class EncoderGateway:
    """ZMQ gateway between PD clients and encoder servers.

    CRITICAL: Uses per-encoder FIFO queues to match responses to clients.
    Encoders MUST process requests in strict FIFO order — concurrent or
    out-of-order processing causes silent response misrouting.
    """

    def __init__(
        self,
        host,
        port,
        encoder_urls,
        scheduler_name=DEFAULT_SCHEDULER,
        stats_interval=10,
    ):
        self._host = host
        self._port = port
        self._encoder_urls = list(encoder_urls)
        self._num_encoders = len(self._encoder_urls)
        self._stats_interval = stats_interval
        self._running = True

        scheduler_cls = SCHEDULER_REGISTRY.get(scheduler_name)
        if scheduler_cls is None:
            raise ValueError(
                f"Unknown scheduler '{scheduler_name}'. "
                f"Available: {list(SCHEDULER_REGISTRY.keys())}"
            )
        self._scheduler = scheduler_cls(self._num_encoders)

        self._ctx = zmq.Context()

        # Frontend ROUTER socket (PD connections)
        self._frontend = self._ctx.socket(zmq.ROUTER)
        self._frontend.setsockopt(zmq.RCVHWM, 0)
        self._frontend.setsockopt(zmq.SNDHWM, 0)
        self._frontend.bind(f"tcp://{host}:{port}")

        # One DEALER per encoder
        self._dealers = []
        self._dealer_queues = []  # FIFO identity queues
        for url in self._encoder_urls:
            dealer = self._ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.RCVHWM, 0)
            dealer.setsockopt(zmq.SNDHWM, 0)
            dealer.connect(url)
            self._dealers.append(dealer)
            self._dealer_queues.append(collections.deque())

        self._last_stats_time = time.monotonic()

        logger.info(
            "EncoderGateway initialized: frontend=tcp://%s:%d, encoders=%s, scheduler=%s",
            host,
            port,
            self._encoder_urls,
            self._scheduler.name,
        )

    # Frame helpers

    @staticmethod
    def _split_identity(frames):
        """Split ROUTER frames into (identity_frames, data_frames).

        ROUTER recv format: [client_id, "", data...]
        """
        delimiter_idx = None
        for i, frame in enumerate(frames):
            if len(frame) == 0:
                delimiter_idx = i
                break
        if delimiter_idx is None:
            raise RuntimeError("No empty delimiter found in ROUTER frames")
        identity_frames = frames[: delimiter_idx + 1]
        data_frames = frames[delimiter_idx + 1 :]
        return identity_frames, data_frames

    # Request / Response handlers

    def _handle_frontend(self):
        """Forward requests from PD clients to encoder backends."""
        while True:
            if not self._frontend.poll(timeout=0):  # drain all ready messages
                break

            frames = self._frontend.recv_multipart(copy=False)
            identity_frames, data_frames = self._split_identity(frames)

            idx = self._scheduler.select(data_frames)

            self._dealer_queues[idx].append(identity_frames)

            # Prepend empty delimiter: DEALER doesn't add it, but encoder ROUTER expects it
            self._dealers[idx].send_multipart(
                [zmq.Frame(b"")] + list(data_frames), copy=False
            )

            self._scheduler.on_request_sent(idx, data_frames)

    def _handle_backend(self, idx):
        """Forward responses from encoder idx back to PD clients.

        Streaming-aware: intermediate messages (with __stream_seq__) peek identity,
        final/non-streaming messages pop identity as before.
        """
        dealer = self._dealers[idx]
        while True:
            if not dealer.poll(timeout=0):
                break

            frames = dealer.recv_multipart(copy=False)

            # Strip leading empty delimiter added by encoder ROUTER
            data_start = 0
            if len(frames) > 0 and len(frames[0]) == 0:
                data_start = 1

            # 判断是否为流式中间消息
            is_intermediate = False
            if data_start < len(frames):
                try:
                    is_intermediate = peek_stream_metadata(frames[data_start])
                except Exception:
                    pass  # 解析失败视为非流式消息

            # Pop client identity (FIFO)
            if not self._dealer_queues[idx]:
                logger.error(
                    "Received unexpected response from encoder %d (empty identity queue), "
                    "dropping %d frames",
                    idx,
                    len(frames),
                )
                continue

            if is_intermediate:
                identity_frames = self._dealer_queues[idx][0]
            else:
                identity_frames = self._dealer_queues[idx].popleft()
                self._scheduler.on_response_received(idx)

            self._frontend.send_multipart(
                list(identity_frames) + list(frames[data_start:]), copy=False
            )

    # Stats

    def _maybe_log_stats(self):
        """Print periodic stats if interval has elapsed."""
        if self._stats_interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats_time < self._stats_interval:
            return
        self._last_stats_time = now
        inflight = [len(q) for q in self._dealer_queues]
        per_enc = ", ".join(
            f"E{i}: inflight={inflight[i]}" for i in range(self._num_encoders)
        )
        extra = self._scheduler.stats_extra()
        logger.info(
            "%s%s",
            per_enc,
            f" | {extra}" if extra else "",
        )

    def _log_final_stats(self):
        """Print final statistics on shutdown."""
        inflight = [len(q) for q in self._dealer_queues]
        logger.info("=== Encoder Gateway Final Stats ===")
        for i in range(self._num_encoders):
            logger.info(
                "  Encoder[%d] (%s): inflight=%d",
                i,
                self._encoder_urls[i],
                inflight[i],
            )

    # Main loop

    def run(self):
        """Run the gateway event loop until SIGINT/SIGTERM."""
        poller = zmq.Poller()
        poller.register(self._frontend, zmq.POLLIN)
        for dealer in self._dealers:
            poller.register(dealer, zmq.POLLIN)

        logger.info("Gateway running — forwarding to %d encoder(s)", self._num_encoders)

        while self._running:
            try:
                events = dict(poller.poll(timeout=1000))
            except zmq.ZMQError:
                if not self._running:
                    break
                raise

            if self._frontend in events:
                self._handle_frontend()

            for i, dealer in enumerate(self._dealers):
                if dealer in events:
                    self._handle_backend(i)

            self._maybe_log_stats()

        self._shutdown()

    def _shutdown(self):
        """Close sockets and terminate context."""
        self._log_final_stats()
        logger.info("Shutting down...")
        self._frontend.close(linger=1000)
        for dealer in self._dealers:
            dealer.close(linger=1000)
        self._ctx.term()
        logger.info("Gateway stopped.")

    def stop(self):
        """Signal the event loop to stop."""
        self._running = False


# CLI entry point


def main():
    parser = argparse.ArgumentParser(
        description="Encoder Gateway — pluggable-scheduler proxy for encoder servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python encoder_gateway.py --port 30010 \\
      --encoder-url tcp://127.0.0.1:30000 tcp://127.0.0.1:30001 \\
      --scheduler data-size-balanced
""",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Bind port for PD connections"
    )
    parser.add_argument(
        "--encoder-url",
        nargs="+",
        required=True,
        dest="encoder_urls",
        help="Encoder server URLs (e.g. tcp://127.0.0.1:30000)",
    )
    parser.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER,
        choices=list(SCHEDULER_REGISTRY.keys()),
        help=f"Scheduling strategy (default: {DEFAULT_SCHEDULER})",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=10,
        help="Seconds between stats logs (0=disable, default=10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gateway = EncoderGateway(
        host=args.host,
        port=args.port,
        encoder_urls=args.encoder_urls,
        scheduler_name=args.scheduler,
        stats_interval=args.stats_interval,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, stopping...", sig_name)
        gateway.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    gateway.run()


if __name__ == "__main__":
    main()
