
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from agentforge.context.models import ContextConfig, ContextPacket

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build a relevance-ranked, budget-aware LLM context.

    Pipeline:
        Gather -> Select -> Structure -> Compress

    memory_tool and rag_tool are optional. Each adapter is expected
    to expose run(dict) or execute(dict), depending on the existing
    AgentForge tool implementation.
    """

    def __init__(
        self,
        memory_tool: Any | None = None,
        rag_tool: Any | None = None,
        config: ContextConfig | None = None,
        history_limit: int = 5,
    ) -> None:
        if history_limit < 0:
            raise ValueError("history_limit cannot be negative")

        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self.history_limit = history_limit
        self.last_stats: dict[str, Any] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def build(
        self,
        user_query: str,
        conversation_history: Sequence[Any] | None = None,
        system_instructions: str | None = None,
        custom_packets: Sequence[ContextPacket] | None = None,
    ) -> str:
        """Run the complete GSSC pipeline and return a context string."""

        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query cannot be empty")

        # G: Collect candidate context from available sources.
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets,
        )

        # S: Select valuable information within the budget.
        selected = self._select(
            packets=packets,
            user_query=user_query,
            available_tokens=self.config.max_tokens,
        )

        # S: Organize selected information into a stable template.
        context = self._structure(
            selected_packets=selected,
            user_query=user_query,
        )

        # C: Reduce the final text if the estimated size is too large.
        if self.config.enable_compression:
            context = self._compress(
                context=context,
                max_tokens=self.config.max_tokens,
            )

        self.last_stats = {
            "candidate_packets": len(packets),
            "selected_packets": len(selected),
            "candidate_tokens": sum(p.token_count for p in packets),
            "selected_packet_tokens": sum(
                p.token_count for p in selected
            ),
            "final_estimated_tokens": self._count_tokens(context),
            "max_tokens": self.config.max_tokens,
        }

        logger.debug("ContextBuilder stats: %s", self.last_stats)
        return context

    # ---------------------------------------------------------
    # G: Gather
    # ---------------------------------------------------------

    def _gather(
        self,
        user_query: str,
        conversation_history: Sequence[Any] | None = None,
        system_instructions: str | None = None,
        custom_packets: Sequence[ContextPacket] | None = None,
    ) -> list[ContextPacket]:
        packets: list[ContextPacket] = []

        if system_instructions and system_instructions.strip():
            packets.append(
                ContextPacket(
                    content=system_instructions.strip(),
                    timestamp=self._now(),
                    token_count=self._count_tokens(system_instructions),
                    relevance_score=1.0,
                    metadata={"type": "system_instruction"},
                )
            )

        # External sources are isolated: a failure in one doesn't
        # prevent us from gathering information from other sources.
        if self.memory_tool is not None:
            try:
                result = self._call_tool(
                    self.memory_tool,
                    {
                        "action": "search",
                        "query": user_query,
                        "limit": 10,
                        "min_importance": 0.3,
                    },
                )
                packets.extend(
                    self._parse_source_results(
                        result,
                        packet_type="memory",
                    )
                )
            except Exception:
                logger.warning(
                    "Memory retrieval failed; continuing without it",
                    exc_info=True,
                )

        if self.rag_tool is not None:
            try:
                result = self._call_tool(
                    self.rag_tool,
                    {
                        "query": user_query,
                        "top_k": 5,
                    },
                )
                packets.extend(
                    self._parse_source_results(
                        result,
                        packet_type="rag_result",
                    )
                )
            except Exception:
                logger.warning(
                    "RAG retrieval failed; continuing without it",
                    exc_info=True,
                )

        # Preserve only the latest N history messages.
        if conversation_history and self.history_limit:
            for message in conversation_history[-self.history_limit:]:
                content = self._get_value(message, "content", "")
                role = self._get_value(message, "role", "unknown")

                if not isinstance(content, str) or not content.strip():
                    continue

                timestamp = self._get_value(
                    message,
                    "timestamp",
                    self._now(),
                )

                packets.append(
                    ContextPacket(
                        content=f"{role}: {content}",
                        timestamp=self._normalize_timestamp(timestamp),
                        token_count=self._count_tokens(content),
                        relevance_score=0.5,
                        metadata={
                            "type": "conversation_history",
                            "role": role,
                        },
                    )
                )

        if custom_packets:
            for packet in custom_packets:
                if not isinstance(packet, ContextPacket):
                    logger.warning(
                        "Ignoring custom context item of type %s",
                        type(packet).__name__,
                    )
                    continue

                packets.append(packet)

        logger.debug("Gathered %d context packets", len(packets))
        return packets

    @staticmethod
    def _call_tool(tool: Any, parameters: dict[str, Any]) -> Any:
        """Call a supported tool entry point without assuming its backend."""

        run = getattr(tool, "run", None)
        if callable(run):
            return run(parameters)

        execute = getattr(tool, "execute", None)
        if callable(execute):
            # Some AgentForge tools use execute(action, **kwargs);
            # others use execute(dict). Try the dictionary form first.
            try:
                return execute(parameters)
            except TypeError:
                action = parameters.get("action")
                remaining = {
                    key: value
                    for key, value in parameters.items()
                    if key != "action"
                }
                if action is not None:
                    return execute(action, **remaining)
                raise

        raise TypeError(
            f"{type(tool).__name__} exposes neither run() nor execute()"
        )

    def _parse_source_results(
        self,
        result: Any,
        packet_type: str,
    ) -> list[ContextPacket]:
        """Normalize common memory/RAG return shapes into packets."""

        if result is None:
            return []

        # A tool may return a list directly or a dictionary wrapping it.
        if isinstance(result, dict):
            items = None

            for key in ("results", "memories", "items", "documents"):
                candidate = result.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break

            if items is None:
                # Some memory tools return a textual summary instead
                # of structured results.
                text = result.get("content") or result.get("message")
                if isinstance(text, str) and text.strip():
                    return [
                        self._make_packet(text, packet_type, result)
                    ]
                return []
        elif isinstance(result, (list, tuple)):
            items = list(result)
        elif isinstance(result, str):
            if not result.strip():
                return []
            return [self._make_packet(result, packet_type)]
        else:
            items = [result]

        packets: list[ContextPacket] = []

        for item in items:
            if isinstance(item, ContextPacket):
                packets.append(item)
                continue

            content = self._get_value(item, "content", None)

            if content is None:
                content = self._get_value(item, "text", None)

            if content is None:
                content = self._get_value(item, "page_content", None)

            if not isinstance(content, str) or not content.strip():
                continue

            metadata = {
                "type": packet_type,
                "source": self._get_value(item, "source", None),
                "page": self._get_value(item, "page", None),
                "id": self._get_value(item, "id", None),
            }
            metadata = {
                key: value
                for key, value in metadata.items()
                if value is not None
            }

            timestamp = self._get_value(
                item,
                "timestamp",
                self._now(),
            )

            # RAG relevance scores may have different scales. Use a
            # conservative default and let the query score the content.
            relevance = self._get_value(
                item,
                "relevance_score",
                self._get_value(item, "score", 0.5),
            )

            try:
                relevance = float(relevance)
            except (TypeError, ValueError):
                relevance = 0.5

            # Scores from some retrieval systems aren't in [0, 1].
            # Clamp them because ContextPacket uses a normalized score.
            relevance = max(0.0, min(1.0, relevance))

            packets.append(
                ContextPacket(
                    content=content.strip(),
                    timestamp=self._normalize_timestamp(timestamp),
                    token_count=self._count_tokens(content),
                    relevance_score=relevance,
                    metadata=metadata,
                )
            )

        return packets

    def _make_packet(
        self,
        content: str,
        packet_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextPacket:
        return ContextPacket(
            content=content.strip(),
            timestamp=self._now(),
            token_count=self._count_tokens(content),
            relevance_score=0.5,
            metadata={
                **(metadata or {}),
                "type": packet_type,
            },
        )

    # ---------------------------------------------------------
    # S: Select
    # ---------------------------------------------------------

    def _select(
        self,
        packets: list[ContextPacket],
        user_query: str,
        available_tokens: int,
    ) -> list[ContextPacket]:
        """Rank packets and select them without exceeding the budget."""

        system_packets = [
            packet
            for packet in packets
            if packet.metadata.get("type") == "system_instruction"
        ]
        other_packets = [
            packet
            for packet in packets
            if packet.metadata.get("type") != "system_instruction"
        ]

        system_tokens = sum(
            self._packet_tokens(packet)
            for packet in system_packets
        )

        # Keep system instructions even if they exceed the budget.
        # Silently deleting safety or role instructions is worse.
        if system_tokens >= available_tokens:
            logger.warning(
                "System instructions use %d estimated tokens; "
                "budget is %d. Other packets will be omitted.",
                system_tokens,
                available_tokens,
            )
            return system_packets

        # reserve_ratio is the portion of the total budget protected
        # for system instructions. The usable non-system budget is
        # the total budget minus the larger of actual and reserved use.
        reserved_tokens = math.ceil(
            available_tokens * self.config.reserve_ratio
        )
        non_system_budget = available_tokens - max(
            system_tokens,
            reserved_tokens,
        )

        scored: list[tuple[float, int, ContextPacket]] = []

        for index, packet in enumerate(other_packets):
            relevance = packet.reveleance_score

            # 0.5 is the model's default/unset score. Recompute it
            # from the query so ordinary packets aren't all tied.
            if relevance == 0.5:
                relevance = self._calculate_relevance(
                    packet.content,
                    user_query,
                )

            if relevance < self.config.min_relevance:
                continue

            recency = self._calculate_recency(packet.timestamp)

            score = (
                self.config.relevance_weight * relevance
                + self.config.recency_weight * recency
            )

            scored.append((score, index, packet))

        # Stable tie-breaking: original order is preserved.
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected = list(system_packets)
        used_non_system_tokens = 0

        for _, _, packet in scored:
            packet_tokens = self._packet_tokens(packet)

            if used_non_system_tokens + packet_tokens > non_system_budget:
                continue

            selected.append(packet)
            used_non_system_tokens += packet_tokens

        return selected

    @staticmethod
    def _calculate_relevance(content: str, query: str) -> float:
        """Calculate Jaccard similarity over normalized word tokens."""

        content_words = set(re.findall(r"\w+", content.casefold()))
        query_words = set(re.findall(r"\w+", query.casefold()))

        if not query_words or not content_words:
            return 0.0

        intersection = len(content_words & query_words)
        union = len(content_words | query_words)

        return intersection / union if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """Exponential decay; newer packets receive higher scores."""

        timestamp = self._normalize_timestamp(timestamp)
        age_hours = max(
            0.0,
            (self._now() - timestamp).total_seconds() / 3600,
        )

        score = math.exp(-0.1 * age_hours / 24)
        return max(0.1, min(1.0, score))

    # ---------------------------------------------------------
    # S: Structure
    # ---------------------------------------------------------

    def _structure(
        self,
        selected_packets: list[ContextPacket],
        user_query: str,
    ) -> str:
        """Build the stable sectioned template from Chapter 9."""

        policies: list[str] = []
        evidence: list[str] = []
        context: list[str] = []
        state: list[str] = []
        output_requirements: list[str] = []

        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")

            if packet_type == "system_instruction":
                policies.append(packet.content)

            elif packet_type in {"rag_result", "knowledge"}:
                source = packet.metadata.get("source")
                page = packet.metadata.get("page")

                label = ""
                if source is not None:
                    label = f"Source: {source}"
                    if page is not None:
                        label += f", page {page}"

                evidence.append(
                    f"{label}\n{packet.content}".strip()
                )

            elif packet_type in {"state", "task_state"}:
                state.append(packet.content)

            elif packet_type == "output_requirements":
                output_requirements.append(packet.content)

            else:
                # Memory, history, and custom context belong here.
                context.append(packet.content)

        sections: list[str] = []

        if policies:
            sections.append(
                "[Role & Policies]\n" + "\n\n".join(policies)
            )

        sections.append("[Task]\n" + user_query.strip())

        if state:
            sections.append("[State]\n" + "\n\n".join(state))

        if evidence:
            sections.append(
                "[Evidence]\n" + "\n\n---\n\n".join(evidence)
            )

        if context:
            sections.append("[Context]\n" + "\n\n".join(context))

        output_text = (
            "\n".join(output_requirements)
            if output_requirements
            else "Provide an accurate, relevant answer. "
                 "Distinguish retrieved evidence from assumptions."
        )
        sections.append("[Output]\n" + output_text)

        return "\n\n".join(sections)

    # ---------------------------------------------------------
    # C: Compress
    # ---------------------------------------------------------

    def _compress(self, context: str, max_tokens: int) -> str:
        """Fallback compression that preserves sections and the task.

        This is deterministic truncation, not semantic summarization.
        A later optional LLM compressor can summarize selected sections.
        """

        if self._count_tokens(context) <= max_tokens:
            return context

        logger.warning(
            "Context exceeds budget; applying deterministic compression"
        )

        sections = context.split("\n\n")
        task_sections = [
            section for section in sections
            if section.startswith("[Task]\n")
        ]
        policy_sections = [
            section for section in sections
            if section.startswith("[Role & Policies]\n")
        ]
        other_sections = [
            section for section in sections
            if not (
                section.startswith("[Task]\n")
                or section.startswith("[Role & Policies]\n")
            )
        ]

        # The system instructions and task are preserved first.
        essential = policy_sections + task_sections
        essential_text = "\n\n".join(essential)

        if self._count_tokens(essential_text) >= max_tokens:
            logger.error(
                "Policies and task alone exceed the estimated context budget"
            )
            # Never silently remove system instructions. The caller
            # should increase the budget or shorten instructions.
            return essential_text

        output_sections = list(essential)
        current_tokens = self._count_tokens(essential_text)

        # Retain sections in original order when there is space.
        for section in other_sections:
            remaining = max_tokens - current_tokens
            section_tokens = self._count_tokens(section)

            if section_tokens <= remaining:
                output_sections.append(section)
                current_tokens += section_tokens
                continue

            # Preserve a useful prefix of the final section if possible.
            truncated = self._truncate_to_tokens(section, remaining)

            if truncated.strip():
                output_sections.append(truncated)

            break

        compressed = "\n\n".join(output_sections)

        # The estimate is approximate; enforce the same estimator
        # again so output doesn't exceed the configured target estimate.
        if self._count_tokens(compressed) > max_tokens:
            compressed = self._truncate_to_tokens(
                compressed,
                max_tokens,
            )

        return compressed

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Fast conservative estimate without an extra tokenizer.

        This is an approximation, not Gemini's official token count.
        Use the provider's tokenizer/count_tokens API for production.
        """

        if not text:
            return 0

        # Roughly 1 token per 4 characters for typical English text.
        return max(1, math.ceil(len(text) / 4))

    def _packet_tokens(self, packet: ContextPacket) -> int:
        # Recalculate missing counts, but preserve explicitly supplied
        # positive token counts from callers.
        return (
            packet.token_count
            if packet.token_count > 0
            else self._count_tokens(packet.content)
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime:
        if not isinstance(value, datetime):
            return datetime.now(timezone.utc)

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _get_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)

        return getattr(item, key, default)
