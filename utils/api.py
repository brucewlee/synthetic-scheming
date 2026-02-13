"""
API utilities with retry logic for rate limit handling.
"""

import time
import random
from typing import Optional, Callable, Any
from openai import OpenAI, RateLimitError, APIError


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 10  # seconds
DEFAULT_MAX_DELAY = 120  # seconds


# ============================================================================
# RETRY WRAPPER
# ============================================================================

def call_with_retry(
    api_call: Callable[[], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    verbose: bool = True
) -> Any:
    """
    Execute an API call with exponential backoff retry on rate limit errors.

    Args:
        api_call: A callable that makes the API request (e.g., lambda: client.chat.completions.create(...))
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        verbose: Whether to print retry messages

    Returns:
        The API response if successful

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return api_call()

        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries:
                break

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)

            if verbose:
                print(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")

            time.sleep(delay)

        except APIError as e:
            # Retry on 5xx server errors
            if e.status_code and e.status_code >= 500:
                last_exception = e
                if attempt == max_retries:
                    break

                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)

                if verbose:
                    print(f"Server error ({e.status_code}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")

                time.sleep(delay)
            else:
                raise

    raise last_exception


def create_chat_completion_with_retry(
    client: OpenAI,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    verbose: bool = True,
    **kwargs
) -> Any:
    """
    Wrapper for client.chat.completions.create() with retry logic.

    Args:
        client: OpenAI client instance
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        verbose: Whether to print retry messages
        **kwargs: Arguments to pass to chat.completions.create()

    Returns:
        The API response if successful
    """
    return call_with_retry(
        api_call=lambda: client.chat.completions.create(**kwargs),
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        verbose=verbose
    )
