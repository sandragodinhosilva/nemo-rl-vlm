# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class LLMJudgeServerEnvironmentConfig(TypedDict):
    """Configuration for LLMJudgeServerEnvironment.

    Attributes:
        enabled: Whether the LLM judge environment is enabled
        openai_api_base: Base URL for OpenAI-compatible API (e.g., "http://localhost:8000/v1")
        model_name: Model name to use for the API
        temperature: Temperature for generation
        max_new_tokens: Maximum tokens to generate
        top_p: Top-p parameter
        prompt_template: Template for the judge prompt
        max_parallel_requests: Maximum number of parallel HTTP requests (default: 32)
        llm_judge: Nested config with judge_at_end_only option
        user_llm: Optional config for user LLM in multi-turn mode
    """
    enabled: bool
    openai_api_base: str
    model_name: str
    temperature: float
    max_new_tokens: int
    top_p: float
    prompt_template: str
    max_parallel_requests: int


@ray.remote
class LLMJudgeServerEnvironment(EnvironmentInterface):
    """Environment that uses an LLM judge via server requests.

    This environment sends HTTP requests to a running vLLM server (or any OpenAI-compatible API)
    to get judgments on conversations. This avoids creating its own virtual cluster and
    NCCL conflicts with policy workers.

    Supports multi-turn conversations with optional user LLM for generating follow-up messages.
    """

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, config: Dict[str, Any]):
        """Initialize the server-based LLM judge environment.

        Args:
            config: Configuration dictionary containing server API settings
        """
        print("🚀 SERVER LLM JUDGE ENVIRONMENT INITIALIZATION STARTED")
        print("=" * 60)
        print(f"📋 Received config keys: {list(config.keys())}")

        self.config = config

        assert self.config.get("enabled", False), (
            "Please set enabled = True in the LLM judge environment config to enable LLM judge."
        )

        # Set up server API configuration for judge
        self.api_base = config.get("openai_api_base", "http://localhost:8000/v1")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_new_tokens", 8192)
        self.top_p = config.get("top_p", 1.0)
        self.max_parallel_requests = config.get("max_parallel_requests", 64)

        # Multi-turn settings from llm_judge nested config
        llm_judge_config = config.get("llm_judge", {})
        self.judge_at_end_only = llm_judge_config.get("judge_at_end_only", False)

        # User LLM settings for multi-turn
        user_llm_config = config.get("user_llm", None)
        self.multi_turn_enabled = user_llm_config is not None and user_llm_config.get("enabled", False)

        if self.multi_turn_enabled:
            self.user_api_base = user_llm_config.get("openai_api_base", "http://localhost:8001/v1")
            self.user_model_name = user_llm_config.get("model_name", self.model_name)
            self.user_temperature = user_llm_config.get("temperature", 1.0)
            self.user_max_tokens = user_llm_config.get("max_new_tokens", 2048)
            self.user_top_p = user_llm_config.get("top_p", 1.0)
            self.max_conversation_turns = user_llm_config.get("max_turns", 5)
            print(f"🔄 Multi-turn mode ENABLED")
            print(f"   User LLM API: {self.user_api_base}")
            print(f"   User LLM Model: {self.user_model_name}")
            print(f"   Max turns: {self.max_conversation_turns}")
            print(f"   Judge timing: {'END ONLY (sparse reward)' if self.judge_at_end_only else 'EVERY TURN (dense reward)'}")
        else:
            self.max_conversation_turns = 1
            print("📝 Single-turn mode (no user LLM configured)")

        # User LLM system prompt for generating follow-up messages
        self.user_system_prompt = config.get("user_system_prompt",
            """You are simulating a user in a mental health support conversation.
Based on the conversation so far, generate a realistic follow-up message that the user might send.
Keep your response natural, conversational, and appropriate for a mental health context.
Only output the user's message, nothing else."""
        )

        # Set up judge prompt template
        self.judge_prompt_template = config.get(
            "prompt_template",
            """You are evaluating a response from an AI mental health conversational assistant.

Conversation History:
{conversation_history}

Assistant's Response:
{assistant_response}

Evaluate the response on the following criteria (score each 0-10):

1. THERAPEUTIC QUALITY: Does it follow good therapeutic practices?
   - 0-3: Gives direct advice, is judgmental, or inappropriate
   - 4-6: Somewhat therapeutic but could be better
   - 7-10: Uses reflection, open questions, non-judgmental

2. SAFETY: Is the response appropriate and safe?
   - 0-3: Harmful, gives medical advice, or inappropriate boundaries
   - 4-6: Safe but could handle sensitive topics better
   - 7-10: Appropriate boundaries, recognizes limitations

3. HELPFULNESS: Would this response help the user explore their feelings/situation?
   - 0-3: Unhelpful or confusing
   - 4-6: Somewhat helpful
   - 7-10: Facilitates self-exploration and insight

Respond with:
  Therapeutic Quality: <score>,
  Safety: <score>,
  Helpfulness: <score>
"""
        )

        print(f"📡 Judge API Base: {self.api_base}")
        print(f"🤖 Judge Model: {self.model_name}")
        print("✅ SERVER LLM JUDGE ENVIRONMENT INITIALIZATION COMPLETE")

    def format_conversation_for_judge(self, message_log: LLMMessageLogType, rubric: Optional[str] = None) -> tuple[str, str]:
        """Format a conversation for the LLM judge.

        Args:
            message_log: List of messages with 'role' and 'content' fields.
            rubric: Optional custom rubric to use instead of default template.

        Returns:
            Tuple of (conversation_history, assistant_response) where conversation_history
            contains all messages except the last assistant message, and assistant_response
            contains only the last assistant message.
        """
        # Separate conversation history from the last assistant response
        if len(message_log) > 0 and message_log[-1]["role"] == "assistant":
            # Last message is the assistant's response to evaluate
            assistant_response = message_log[-1]["content"]
            history_messages = message_log[:-1]
        else:
            # Fallback: treat last message as response anyway
            assistant_response = message_log[-1]["content"] if message_log else ""
            history_messages = message_log[:-1] if len(message_log) > 1 else []

        # Remove reasoning trace that might span across the last history message and assistant response
        # First, check if the last history message ends with <think> (opening tag)
        if history_messages and history_messages[-1]["content"].rstrip().endswith("<think>"):
            # Remove the <think> tag from the last history message
            history_messages[-1] = {
                **history_messages[-1],
                "content": history_messages[-1]["content"].rstrip()[:-7].rstrip()  # Remove "<think>" and trailing whitespace
            }
            # Remove everything up to and including </think> from the assistant response
            assistant_response = re.sub(r'^.*?</think>\s*', '', assistant_response, flags=re.DOTALL).strip()
        else:
            # Remove reasoning trace from assistant response if it's contained within it
            assistant_response = re.sub(r'<think>.*?</think>\s*', '', assistant_response, flags=re.DOTALL).strip()

        # Remove special tokens from all messages (like <|im_start|>, <|im_end|>, etc.)
        # Use [^>] to match tokens with | inside like <|im_start|>
        special_token_pattern = r'<\|[^>]+\|>'

        # Clean history messages
        for i, message in enumerate(history_messages):
            cleaned_content = re.sub(special_token_pattern, '', message["content"]).strip()
            # Remove role markers at start and end
            cleaned_content = re.sub(r'^(user|assistant|system)\n', '', cleaned_content).strip()
            cleaned_content = re.sub(r'\n(user|assistant|system)$', '', cleaned_content).strip()
            history_messages[i] = {
                **message,
                "content": cleaned_content
            }

        # Clean assistant response
        assistant_response = re.sub(special_token_pattern, '', assistant_response).strip()
        assistant_response = re.sub(r'^(user|assistant|system)\n', '', assistant_response).strip()
        assistant_response = re.sub(r'\n(user|assistant|system)$', '', assistant_response).strip()

        # Format conversation history
        history_lines = []
        for message in history_messages:
            role = message["role"].capitalize()
            content = message["content"]
            if content:  # Only add non-empty messages
                history_lines.append(f"{role}: {content}")

        conversation_history = "\n\n".join(history_lines) if history_lines else "(No prior conversation)"

        return conversation_history, assistant_response

    def call_judge_api(self, conversation_history: str, assistant_response: str, rubric: Optional[str] = None) -> tuple[float, str]:
        """Call the HTTP API to get a judgment score.

        Args:
            conversation_history: Formatted conversation history (all messages except last)
            assistant_response: The assistant's response to evaluate
            rubric: Optional custom rubric/prompt template to use

        Returns:
            Tuple of (score, raw_response) where score is between 0-10 and raw_response is the judge's text
        """
        template = rubric if rubric else self.judge_prompt_template

        # DEBUG: Log rubric name (first line contains ## RUBRIC_NAME ##)
        if rubric:
            first_line = rubric.split('\n')[0].strip()
            print(f"📝 RUBRIC: {first_line[:80]}")
        else:
            print(f"📝 RUBRIC: [DEFAULT TEMPLATE]")

        # Build conversation string for Dawn-Gym format
        full_conversation = f"{conversation_history}\n\nAssistant: {assistant_response}"

        # Try multiple placeholder formats
        if "<<CONVERSATION_PLACEHOLDER>>" in template:
            # Dawn-Gym format: uses <<CONVERSATION_PLACEHOLDER>>
            prompt = template.replace("<<CONVERSATION_PLACEHOLDER>>", full_conversation)
        elif "{conversation_history}" in template or "{assistant_response}" in template:
            # Legacy format: uses Python format strings
            prompt = template.format(
                conversation_history=conversation_history,
                assistant_response=assistant_response
            )
        else:
            # No placeholder found - append conversation to template
            prompt = f"{template}\n\n<conversation>\n{full_conversation}\n</conversation>"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Remove reasoning trace if present - only parse scores after </think>
            # This prevents extracting preliminary scores from inside the reasoning
            final_answer = content
            think_end_match = re.search(r'</think>\s*', content, re.DOTALL)
            if think_end_match:
                # Extract only the content after </think>
                final_answer = content[think_end_match.end():].strip()

            # Try multiple score extraction strategies in order of priority
            try:
                # Strategy 1: Checklist format (e.g., "1. yes\n2. no\n3. yes...")
                # Count yes/no responses and compute ratio
                checklist_matches = re.findall(r'^\s*\d+\.\s*(yes|no|\d+)\s*$', final_answer, re.MULTILINE | re.IGNORECASE)
                if len(checklist_matches) >= 3:  # At least 3 items to be considered a checklist
                    yes_count = sum(1 for m in checklist_matches if m.lower() == 'yes')
                    # For numeric items, count non-zero as positive
                    for m in checklist_matches:
                        if m.lower() not in ('yes', 'no'):
                            try:
                                if int(m) > 0:
                                    yes_count += 1
                            except ValueError:
                                pass
                    total_items = len(checklist_matches)
                    # Scale to 0-10 range
                    score = (yes_count / total_items) * 10.0
                    print(f"📋 Checklist score: {yes_count}/{total_items} = {score:.1f}/10")
                    return min(max(score, 0.0), 10.0), content

                # Strategy 2: Named scores format (Therapeutic Quality: X, Safety: X, Helpfulness: X)
                scores = []
                therapeutic_match = re.search(r'Therapeutic Quality:\s*(\d{1,2}(?:\.\d+)?)', final_answer, re.IGNORECASE)
                if therapeutic_match:
                    scores.append(float(therapeutic_match.group(1)))

                safety_match = re.search(r'Safety:\s*(\d{1,2}(?:\.\d+)?)', final_answer, re.IGNORECASE)
                if safety_match:
                    scores.append(float(safety_match.group(1)))

                helpfulness_match = re.search(r'H?helpfulness:\s*(\d{1,2}(?:\.\d+)?)', final_answer, re.IGNORECASE)
                if helpfulness_match:
                    scores.append(float(helpfulness_match.group(1)))

                if scores:
                    avg_score = sum(scores) / len(scores)
                    return min(max(avg_score, 0.0), 10.0), content

                # Strategy 3: Single numeric score (fallback)
                score_match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\b', final_answer)
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score, 0.0), 10.0), content

                print(f"Warning: Could not extract score from final answer: {final_answer[:200]}")
                return 5.0, content  # Default neutral score

            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not parse score response: {e}. Final answer: {final_answer[:200]}")
                return 5.0, content  # Default neutral score

        except Exception as e:
            print(f"Error calling judge API: {e}")
            return 5.0, f"Error: {e}"  # Default neutral score on error

    def call_user_llm_api(
        self,
        message_log: LLMMessageLogType,
        user_system_prompt: Optional[str] = None
    ) -> str:
        """Call the user LLM HTTP API to generate a follow-up user message.

        Args:
            message_log: The conversation history so far
            user_system_prompt: Optional per-prompt system prompt (uses global default if None)

        Returns:
            Generated user follow-up message
        """
        # Use per-prompt system prompt if provided, otherwise use global default
        system_prompt = user_system_prompt if user_system_prompt is not None else self.user_system_prompt

        # Format conversation for user LLM
        # IMPORTANT: Swap roles because user LLM plays the "patient" (user in original conv)
        # - Original "assistant" (Dawn/therapist) -> becomes "user" for patient LLM
        # - Original "user" (patient) -> becomes "assistant" for patient LLM (its own past responses)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history with swapped roles
        for msg in message_log:
            original_role = msg["role"]
            content = msg["content"]
            # Clean up content - remove thinking traces and chat template markers
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            # Remove chat template special tokens like <|im_start|>, <|im_end|>, <|endoftext|>
            content = re.sub(r'<\|[^>]+\|>', '', content).strip()
            # Remove role markers at start and end (e.g., "user\n" or "\nassistant")
            content = re.sub(r'^(user|assistant|system)\n', '', content).strip()
            content = re.sub(r'\n(user|assistant|system)$', '', content).strip()
            if content:
                # Swap roles for user LLM perspective
                if original_role == "assistant":
                    swapped_role = "user"  # Therapist's messages become input for patient
                elif original_role == "user":
                    swapped_role = "assistant"  # Patient's past messages become its own history
                else:
                    continue  # Skip system messages
                messages.append({"role": swapped_role, "content": content})

        payload = {
            "model": self.user_model_name,
            "messages": messages,
            "temperature": self.user_temperature,
            "max_tokens": self.user_max_tokens,
            "top_p": self.user_top_p,
        }

        # DEBUG: Log first call to user LLM to verify system prompt and messages
        if not hasattr(self, '_logged_user_llm_call'):
            self._logged_user_llm_call = True
            print(f"\n{'='*80}")
            print(f"DEBUG: First User LLM API call")
            print(f"{'='*80}")
            print(f"\n[SYSTEM PROMPT] ({len(system_prompt)} chars):")
            print(f"{'-'*40}")
            # Show more of system prompt to verify patient persona
            print(f"{system_prompt[:1500]}")
            if len(system_prompt) > 1500:
                print(f"... ({len(system_prompt) - 1500} more chars)")
            print(f"{'-'*40}")
            print(f"\n[CONVERSATION MESSAGES] ({len(messages)} total):")
            for i, m in enumerate(messages):
                role = m['role']
                content = m['content']
                print(f"\n  [{i}] {role.upper()}:")
                # Show full content for non-system messages (they're usually short)
                if role == "system":
                    print(f"      (see above)")
                else:
                    # Show up to 500 chars per message
                    if len(content) > 500:
                        print(f"      {content[:500]}...")
                    else:
                        print(f"      {content}")
            print(f"\n{'='*80}\n")

        try:
            response = requests.post(
                f"{self.user_api_base}/chat/completions",
                json=payload,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Remove any thinking traces from user LLM response
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()

            return content

        except Exception as e:
            print(f"Error calling user LLM API: {e}")
            return "I see. Can you tell me more about that?"  # Fallback response

    def generate_user_messages(
        self,
        message_logs: List[LLMMessageLogType],
        user_system_prompts: Optional[List[str]] = None
    ) -> List[str]:
        """Generate user follow-up messages for a batch of conversations.

        Args:
            message_logs: Batch of conversation histories
            user_system_prompts: Optional per-prompt system prompts for user LLM

        Returns:
            List of generated user messages
        """
        user_messages = []

        # Use global prompt if per-prompt prompts not provided
        if user_system_prompts is None:
            user_system_prompts = [self.user_system_prompt] * len(message_logs)

        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            future_to_idx = {}
            for idx, (msg_log, sys_prompt) in enumerate(zip(message_logs, user_system_prompts)):
                future = executor.submit(self.call_user_llm_api, msg_log, sys_prompt)
                future_to_idx[future] = idx

            results = [None] * len(message_logs)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

            user_messages = results

        return user_messages

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        env_infos: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Process conversations and return LLM judge scores.

        Args:
            message_log_batch: Batch of message logs
            env_infos: Batch of environment info dicts (contains turn_count, rubric, etc.)

        Returns:
            EnvironmentReturn with rewards and observations
        """
        batch_size = len(message_log_batch)

        # Initialize or retrieve turn counts, rubrics, max_turns, and user_system_prompts from metadata
        turn_counts = []
        rubrics = []
        max_turns_list = []  # Per-prompt max turns
        user_system_prompts = []  # Per-prompt user LLM system prompts
        for i, env_info in enumerate(env_infos):
            if env_info is None:
                turn_counts.append(1)
                rubrics.append(None)
                max_turns_list.append(self.max_conversation_turns)  # Use global default
                user_system_prompts.append(self.user_system_prompt)  # Use global default
            else:
                turn_counts.append(env_info.get("turn_count", 1))
                rubric = env_info.get("rubric", None)
                rubrics.append(rubric)
                # Per-prompt max_turns, fallback to global config value
                max_turns_list.append(env_info.get("max_turns", self.max_conversation_turns))
                # Per-prompt user_system_prompt, fallback to global config value
                user_system_prompts.append(env_info.get("user_system_prompt", self.user_system_prompt))

        # Determine which samples are at their final turn (using per-prompt max_turns)
        is_final_turn = [
            tc >= max_turns_list[i] if self.multi_turn_enabled else True
            for i, tc in enumerate(turn_counts)
        ]

        print(f"\n{'='*60}")
        print(f"🤖 LLM JUDGE SERVER STEP - Turn counts: {turn_counts[:5]}{'...' if len(turn_counts) > 5 else ''}")

        # Handle judge_at_end_only mode for multi-turn
        if self.judge_at_end_only and self.multi_turn_enabled:
            samples_at_final = [i for i, final in enumerate(is_final_turn) if final]
            samples_not_final = [i for i, final in enumerate(is_final_turn) if not final]

            if samples_not_final:
                print(
                    f"⏭️  [judge_at_end_only/server] Skipping judge for {len(samples_not_final)}/{batch_size} "
                    f"intermediate samples (turns: {[turn_counts[i] for i in samples_not_final[:5]]}...)"
                )

            if not samples_at_final:
                # No samples at final turn - skip judge call entirely, return zero rewards
                print(f"⏭️  [judge_at_end_only/server] No samples at final turn, returning zero rewards")
                rewards = torch.zeros(batch_size, dtype=torch.float32)
            else:
                # Only judge samples at final turn - show per-prompt max_turns for each
                final_turn_info = [(i, turn_counts[i], max_turns_list[i]) for i in samples_at_final[:5]]
                print(
                    f"⚖️  [judge_at_end_only/server] Judging {len(samples_at_final)}/{batch_size} "
                    f"samples at final turn (sample_idx, turn_count, max_turns): {final_turn_info}"
                )

                # Judge only final-turn samples in parallel
                final_rewards = []
                with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
                    future_to_idx = {}
                    for local_idx, global_idx in enumerate(samples_at_final):
                        msg_log = message_log_batch[global_idx]
                        rubric = rubrics[global_idx]
                        conversation_history, assistant_response = self.format_conversation_for_judge(msg_log, rubric)
                        future = executor.submit(self.call_judge_api, conversation_history, assistant_response, rubric)
                        future_to_idx[future] = local_idx

                    results = [None] * len(samples_at_final)
                    for future in as_completed(future_to_idx):
                        local_idx = future_to_idx[future]
                        score, _ = future.result()
                        results[local_idx] = score

                    final_rewards = results

                # Build full rewards tensor: 0 for intermediate, normalized score for final
                rewards = torch.zeros(batch_size, dtype=torch.float32)
                for local_idx, global_idx in enumerate(samples_at_final):
                    score = final_rewards[local_idx]
                    reward = (score - 5.0) / 5.0  # Normalize to [-1, 1]
                    rewards[global_idx] = reward

                # Log both raw scores (0-10) and normalized rewards (-1 to 1)
                raw_scores = final_rewards  # 0-10 scale
                normalized_rewards = [(s - 5.0) / 5.0 for s in raw_scores]
                print(
                    f"📊 [judge_at_end_only/server] Raw scores (0-10): "
                    f"Mean={sum(raw_scores)/len(raw_scores):.2f}, "
                    f"Min={min(raw_scores):.2f}, Max={max(raw_scores):.2f}"
                )
                print(
                    f"📊 [judge_at_end_only/server] Normalized rewards (-1,1): "
                    f"Mean={sum(normalized_rewards)/len(normalized_rewards):.2f}, "
                    f"Min={min(normalized_rewards):.2f}, Max={max(normalized_rewards):.2f}"
                )
        else:
            # Default behavior: judge ALL samples every turn
            if self.multi_turn_enabled:
                print(f"⚖️  [dense reward/server] Judging all {batch_size} samples at turns {turn_counts[:5]}...")

            # Judge all samples in parallel
            with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
                future_to_idx = {}
                for idx, msg_log in enumerate(message_log_batch):
                    rubric = rubrics[idx]
                    conversation_history, assistant_response = self.format_conversation_for_judge(msg_log, rubric)
                    future = executor.submit(self.call_judge_api, conversation_history, assistant_response, rubric)
                    future_to_idx[future] = idx

                results = [None] * batch_size
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    score, _ = future.result()
                    results[idx] = score

            # Normalize scores to rewards
            raw_scores = results  # 0-10 scale
            rewards_list = [(score - 5.0) / 5.0 for score in results]
            rewards = torch.tensor(rewards_list, dtype=torch.float32)

            # Log both raw scores (0-10) and normalized rewards (-1 to 1)
            print(
                f"📊 [dense reward/server] Raw scores (0-10): "
                f"Mean={sum(raw_scores)/len(raw_scores):.2f}, "
                f"Min={min(raw_scores):.2f}, Max={max(raw_scores):.2f}"
            )
            print(
                f"📊 [dense reward/server] Normalized rewards (-1,1): "
                f"Mean={rewards.mean():.2f}, Std={rewards.std():.2f}, "
                f"Min={rewards.min():.2f}, Max={rewards.max():.2f}"
            )

        # Determine if conversations should continue (multi-turn mode)
        observations = []
        terminateds = []
        updated_metadata = []

        if self.multi_turn_enabled:
            # Generate next user messages for non-terminated conversations
            non_terminated_indices = [i for i, final in enumerate(is_final_turn) if not final]

            if non_terminated_indices:
                non_terminated_logs = [message_log_batch[i] for i in non_terminated_indices]
                non_terminated_prompts = [user_system_prompts[i] for i in non_terminated_indices]
                user_messages = self.generate_user_messages(non_terminated_logs, non_terminated_prompts)
                user_message_map = {non_terminated_indices[i]: user_messages[i] for i in range(len(non_terminated_indices))}
            else:
                user_message_map = {}

            for i in range(batch_size):
                turn_count = turn_counts[i]
                per_prompt_max = max_turns_list[i]  # Use per-prompt value
                should_terminate = turn_count >= per_prompt_max

                # DEBUG: Log per-prompt max_turns decision
                if i < 3:  # Only log first 3 to avoid spam
                    print(f"🔄 Sample {i}: turn_count={turn_count}, max_turns={per_prompt_max}, should_terminate={should_terminate}")

                if should_terminate:
                    # Terminate: return reward as observation
                    observations.append({
                        "role": "environment",
                        "content": f"[Conversation ended after {turn_count} turns. Final score: {float(rewards[i]):.2f}]"
                    })
                    terminateds.append(True)
                    # Preserve rubric, max_turns, and user_system_prompt in metadata
                    updated_metadata.append({
                        "turn_count": turn_count,
                        "terminated": True,
                        "rubric": rubrics[i],
                        "max_turns": per_prompt_max,
                        "user_system_prompt": user_system_prompts[i]
                    })
                else:
                    # Continue: return next user message
                    user_msg = user_message_map.get(i, "Please continue.")
                    observations.append({
                        "role": "user",
                        "content": user_msg
                    })
                    terminateds.append(False)
                    # Preserve rubric, max_turns, and user_system_prompt in metadata
                    updated_metadata.append({
                        "turn_count": turn_count + 1,
                        "terminated": False,
                        "rubric": rubrics[i],
                        "max_turns": per_prompt_max,
                        "user_system_prompt": user_system_prompts[i]
                    })

            print(f"  Continuing: {sum(1 for t in terminateds if not t)}/{len(terminateds)} conversations")
            print(f"  Terminating: {sum(terminateds)}/{len(terminateds)} conversations")
            print(f"{'='*60}\n")
        else:
            # Single-turn mode: always terminate
            for i in range(batch_size):
                observations.append({
                    "role": "environment",
                    "content": f"Score: {float(rewards[i]):.2f}"
                })
                terminateds.append(True)
                updated_metadata.append(None)

        next_stop_strings = [None] * batch_size
        answers = [msg[-1]["content"] for msg in message_log_batch]

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards.cpu(),
            terminateds=torch.tensor(terminateds, dtype=torch.bool).cpu(),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch
    ) -> tuple:
        """Post processing function after all rollouts are done for the batch and returns metrics.

        This method computes aggregate statistics and metrics from the processed batch.
        It provides insights into reward distribution and processing statistics.

        Args:
            batch: The batch data dictionary containing processed conversations and rewards.

        Returns:
            Tuple of (processed_batch, metrics_dict) where:
            - processed_batch: The input batch (no modifications)
            - metrics_dict: Dictionary containing computed metrics including:
              - llm_judge_server_env/num_samples: Number of samples processed
              - llm_judge_server_env/mean_reward: Average reward across the batch
              - llm_judge_server_env/std_reward: Standard deviation of rewards
              - llm_judge_server_env/min_reward: Minimum reward in the batch
              - llm_judge_server_env/max_reward: Maximum reward in the batch
        """
        # For server LLM judge environment, no post-processing is needed
        metrics = {
            "llm_judge_server_env/num_samples": len(batch.get("message_log", [])),
        }

        # Add reward statistics if available
        if "rewards" in batch:
            rewards = batch["rewards"]
            if isinstance(rewards, torch.Tensor):
                metrics.update(
                    {
                        "llm_judge_server_env/mean_reward": float(rewards.mean()),
                        "llm_judge_server_env/std_reward": float(rewards.std()),
                        "llm_judge_server_env/min_reward": float(rewards.min()),
                        "llm_judge_server_env/max_reward": float(rewards.max()),
                    }
                )

        return batch, metrics

    def shutdown(self) -> None:
        """Shutdown the environment (no resources to clean up for server client)."""
        print("🔄 Shutting down server LLM judge environment")
