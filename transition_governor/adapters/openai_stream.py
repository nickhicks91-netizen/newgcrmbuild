"""
OpenAI streaming API adapter.

This adapter integrates the Transition Governor with OpenAI's streaming API.
Since we don't have access to logits, we use proxy signals (logprobs, token variance).
"""

import time
from typing import Optional, AsyncIterator, Iterator, Dict, Any
from dataclasses import dataclass

from ..core.governor import TransitionGovernor
from ..core.state import AIState, ToolState, GovernanceState
from ..core.metrics import ProxyEntropyAdapter, ConfidenceEstimator


@dataclass
class StreamToken:
    """
    Single token from streaming response.

    Contains governor-relevant information extracted from the stream.
    """
    text: str
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[list] = None


class GovernedStreamWrapper:
    """
    Wraps OpenAI streaming responses with governor control.

    This wrapper:
    1. Monitors stream for stability signals (logprobs)
    2. Builds AIState from proxies
    3. Calls governor at each token
    4. Blocks/modifies stream if in brownout
    5. Injects warnings/explanations as needed

    Usage:
        wrapper = GovernedStreamWrapper(governor)
        async for token in wrapper.wrap_stream(openai_stream):
            yield token
    """

    def __init__(
        self,
        governor: TransitionGovernor,
        tool_state_provider: Optional['ToolStateProvider'] = None,
        max_context_length: int = 4096,
        seed: int = 42
    ):
        """
        Initialize wrapper.

        Args:
            governor: TransitionGovernor instance
            tool_state_provider: Provider for tool state (optional)
            max_context_length: Maximum context window
            seed: Random seed
        """
        self.governor = governor
        self.tool_state_provider = tool_state_provider
        self.max_context_length = max_context_length
        self.seed = seed

        # Initialize entropy estimator (uses proxies)
        self.entropy_adapter = ProxyEntropyAdapter(seed=seed)
        self.confidence_estimator = ConfidenceEstimator()

        # Token counter
        self.token_count = 0
        self.start_time = time.time()

    async def wrap_stream(
        self,
        stream: AsyncIterator[Dict[str, Any]],
        context_length: int = 0
    ) -> AsyncIterator[str]:
        """
        Wrap async streaming response with governance.

        Args:
            stream: OpenAI stream iterator
            context_length: Current context length

        Yields:
            Governed tokens (may be modified or blocked)
        """
        brownout_warning_sent = False

        async for chunk in stream:
            # Extract token data
            token_data = self._extract_token_data(chunk)

            if token_data is None:
                continue

            # Build AIState
            state = self._build_state(token_data, context_length)

            # Call governor
            output = self.governor.govern(state)

            # Check for brownout
            if output.governance_state == GovernanceState.BROWNOUT:
                if not brownout_warning_sent:
                    # Send brownout warning once
                    yield "\n[System entering reduced-confidence mode due to instability]\n"
                    brownout_warning_sent = True

                # In brownout, only allow continuation if critical
                # For now, block all tool calls and limit tokens
                if token_data.tool_calls:
                    yield "[Tool calls blocked during brownout]\n"
                    continue

            # Check token limit
            if output.max_tokens_per_step is not None:
                if self.token_count >= output.max_tokens_per_step:
                    yield "\n[Output truncated due to governor limits]\n"
                    break

            # Yield token
            yield token_data.text
            self.token_count += 1

            # Check for finish
            if token_data.finish_reason:
                break

    def wrap_stream_sync(
        self,
        stream: Iterator[Dict[str, Any]],
        context_length: int = 0
    ) -> Iterator[str]:
        """
        Synchronous version of wrap_stream.

        Args:
            stream: OpenAI stream iterator (sync)
            context_length: Current context length

        Yields:
            Governed tokens
        """
        brownout_warning_sent = False

        for chunk in stream:
            # Extract token data
            token_data = self._extract_token_data(chunk)

            if token_data is None:
                continue

            # Build AIState
            state = self._build_state(token_data, context_length)

            # Call governor
            output = self.governor.govern(state)

            # Check for brownout
            if output.governance_state == GovernanceState.BROWNOUT:
                if not brownout_warning_sent:
                    yield "\n[System entering reduced-confidence mode due to instability]\n"
                    brownout_warning_sent = True

                if token_data.tool_calls:
                    yield "[Tool calls blocked during brownout]\n"
                    continue

            # Check token limit
            if output.max_tokens_per_step is not None:
                if self.token_count >= output.max_tokens_per_step:
                    yield "\n[Output truncated due to governor limits]\n"
                    break

            # Yield token
            yield token_data.text
            self.token_count += 1

            if token_data.finish_reason:
                break

    def _extract_token_data(self, chunk: Dict[str, Any]) -> Optional[StreamToken]:
        """
        Extract relevant data from stream chunk.

        Args:
            chunk: Raw chunk from OpenAI stream

        Returns:
            StreamToken or None if chunk is empty
        """
        # OpenAI stream format varies; this is a simplified extraction
        # Real implementation would handle specific API format

        if 'choices' not in chunk:
            return None

        choice = chunk['choices'][0]
        delta = choice.get('delta', {})

        if not delta:
            return None

        text = delta.get('content', '')
        tool_calls = delta.get('tool_calls')
        finish_reason = choice.get('finish_reason')

        # Extract logprob if available
        logprob = None
        if 'logprobs' in choice:
            logprobs = choice['logprobs']
            if logprobs and 'content' in logprobs:
                content_logprobs = logprobs['content']
                if content_logprobs:
                    logprob = content_logprobs[0].get('logprob')

        return StreamToken(
            text=text,
            logprob=logprob,
            finish_reason=finish_reason,
            tool_calls=tool_calls
        )

    def _build_state(
        self,
        token_data: StreamToken,
        context_length: int
    ) -> AIState:
        """
        Build AIState from token data.

        Args:
            token_data: Extracted token information
            context_length: Current context length

        Returns:
            AIState for governance
        """
        # Estimate entropy from logprob (if available)
        if token_data.logprob is not None:
            entropy = self.entropy_adapter.get_entropy(logprob=token_data.logprob)
        else:
            # Fallback: assume moderate entropy
            entropy = 1.0

        entropy_dot = self.entropy_adapter.get_entropy_dot()

        # Compute confidence
        confidence = self.confidence_estimator.estimate_confidence(entropy)
        confidence_dot = self.confidence_estimator.estimate_confidence_dot(
            entropy_dot, confidence
        )

        # Get tool state
        tool_state = (
            self.tool_state_provider.current_state()
            if self.tool_state_provider
            else ToolState.INACTIVE
        )

        # Detect tool calls
        if token_data.tool_calls:
            tool_state = ToolState.PENDING

        # Build state
        state = AIState(
            entropy=entropy,
            entropy_dot=entropy_dot,
            confidence=confidence,
            confidence_dot=confidence_dot,
            tool_state=tool_state,
            context_length=context_length + self.token_count,
            max_context_length=self.max_context_length,
            fatigue=self.governor.fatigue_accumulator.get_fatigue()
        )

        return state

    def reset(self) -> None:
        """Reset wrapper state."""
        self.token_count = 0
        self.start_time = time.time()
        self.governor.reset()
