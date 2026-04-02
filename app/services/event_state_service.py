from __future__ import annotations

from app.domain.entities import DetectionRecord, EventState
from app.domain.enums import EventType


class EventStateService:
    def __init__(
        self,
        trigger_seconds_by_event: dict[EventType, int],
        min_positive_frames_by_event: dict[EventType, int],
        max_negative_frames_by_event: dict[EventType, int],
    ) -> None:
        self.trigger_seconds_by_event = trigger_seconds_by_event
        self.min_positive_frames_by_event = min_positive_frames_by_event
        self.max_negative_frames_by_event = max_negative_frames_by_event
        self.states: dict[tuple[str, str, str], EventState] = {}

    def _key(self, source_id: str, event_type: EventType, entity_id: str) -> tuple[str, str, str]:
        return (source_id, event_type.value, entity_id)

    def update(self, source_id: str, entity_id: str, record: DetectionRecord) -> EventState:
        key = self._key(source_id, record.event_type, entity_id)
        state = self.states.get(key)
        if state is None:
            state = EventState(
                event_type=record.event_type,
                source_id=source_id,
                entity_id=entity_id,
            )
            self.states[key] = state

        if record.event_detected:
            state.positive_counter += 1
            state.negative_counter = 0
            if state.positive_counter >= self.min_positive_frames_by_event[record.event_type]:
                if not state.confirmed_active:
                    state.confirmed_active = True
                    state.active_start_time_seconds = record.timestamp_seconds
        else:
            state.negative_counter += 1
            if state.negative_counter >= self.max_negative_frames_by_event[record.event_type]:
                state.positive_counter = 0
                state.negative_counter = 0
                state.confirmed_active = False
                state.active_start_time_seconds = 0.0
                state.last_api_sent_time_seconds = 0.0
        return state

    def should_notify(self, state: EventState, current_time_seconds: float) -> tuple[bool, float]:
        if not state.confirmed_active:
            return False, 0.0

        duration = current_time_seconds - state.active_start_time_seconds
        trigger_seconds = self.trigger_seconds_by_event[state.event_type]
        elapsed_since_last_api = current_time_seconds - state.last_api_sent_time_seconds

        if duration >= trigger_seconds and elapsed_since_last_api >= trigger_seconds:
            return True, duration
        return False, duration

    def mark_notified(self, state: EventState, current_time_seconds: float) -> None:
        state.last_api_sent_time_seconds = current_time_seconds
