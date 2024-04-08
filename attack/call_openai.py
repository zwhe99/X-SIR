import os       # for reading API key
import re       # for matching endpoint from request URL
import sys      # for reconfiguring stdout and stderr
import json     # for saving results
import time     # for sleeping after rate limit is hit
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import logging  # for logging rate limit warnings and other messages
import tiktoken # for counting tokens

from tqdm import tqdm
from typing import Callable
from const import MODEL2RPM, MODEL2TPM
from dataclasses import (dataclass, field) # for storing API inputs, outputs, and metadata

# reconfigure stdout and stderr to be line-buffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    model: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.encoding_for_model(model)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_total: int = 0
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    response_to_output_func: Callable[[dict, str], None]
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        progress_bar = tqdm
    ):
        """Calls the OpenAI API and saves results."""
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left > 0:
                logging.warning(
                    f"Request {self.request_json} failed with errors: {self.result}. Retry attempt {self.attempts_left} left."
                )
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts."
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = {
                "response": response,
                "metadata": self.metadata if self.metadata else None
            }
            self.response_to_output_func(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            progress_bar.update(n=1)
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

class CallOpenAI:
    def __init__(
        self,
        request_url,
        api_key,
        input_file_path,
        output_file_path,
        input_to_requests_func,
        response_to_output_func,
        is_all_done_func=None,
        post_run_func=None,
        max_attempts=5,
        seconds_to_pause_after_rate_limit_error=15,
        seconds_to_sleep_each_loop=0.001,
        progress_bar_desc=None,
        logging_level=logging.INFO
    ):
        self.request_url = request_url
        self.api_key = api_key
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_to_requests_func = input_to_requests_func
        self.response_to_output_func = response_to_output_func
        self.is_all_done_func = is_all_done_func
        self.post_run_func = post_run_func
        self.api_endpoint = api_endpoint_from_url(request_url)
        self.max_attempts = max_attempts
        self.seconds_to_pause_after_rate_limit_error = seconds_to_pause_after_rate_limit_error
        self.seconds_to_sleep_each_loop = seconds_to_sleep_each_loop
        self.logging_level = logging_level
        self.progress_bar = tqdm(desc=progress_bar_desc)

        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
        if "/deployments" in self.request_url:
            # use api-key header for Azure deployments
            self.request_header = {"api-key": f"{self.api_key}"}

        # initialize logging
        logging.basicConfig(level=logging_level)
        logging.debug(f"Logging initialized at level {logging_level}")

        # initialize trackers
        self.queue_of_requests_to_retry = asyncio.Queue()
        self.task_id_generator = (
            task_id_generator_function()
        )  # generates integer IDs of 0, 1, 2, ...
        self.status_tracker = (
            StatusTracker()
        )  # single instance to track a collection of variables
        self.next_request = None  # variable to hold the next request to call

        # initialize available capacity counts
        self.model = None
        self.max_requests_per_minute = None
        self.max_tokens_per_minute = None
        self.available_request_capacity = None
        self.available_token_capacity = None
        self.last_update_time = time.time()

        # initialize flags
        self.file_not_finished = True  # after file is empty, we'll skip reading it

        # Check input & output file
        assert os.path.isfile(self.input_file_path), f"Input file {self.input_file_path} does NOT exist or is a dir."

        output_directory = os.path.dirname(self.output_file_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        logging.debug(f"Initialization complete.")

    def set_input_to_requests_func(self, input_to_requests_func):
        self.input_to_requests_func = input_to_requests_func

    def set_response_to_output_func(self, response_to_output_func):
        self.response_to_output_func = response_to_output_func

    async def run(self):
        if self.is_all_done_func is not None and self.is_all_done_func(self.input_file_path, self.output_file_path):
            logging.info("All done!")
            return

        requests = self.input_to_requests_func(self.input_file_path, self.output_file_path)

        # set progress bar
        total = len(requests)
        self.progress_bar.reset(total=total)
        self.status_tracker.num_tasks_total = total

        if total == 0:
            logging.info("No requests to run.")
            return

        # set iterator
        requests = iter(requests)

        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if self.next_request is None:
                    if not self.queue_of_requests_to_retry.empty():
                        self.next_request = self.queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {self.next_request.task_id}: {self.next_request}"
                        )
                    elif self.file_not_finished:
                        try:
                            # get new request
                            request_json = next(requests)
                            assert "model" in request_json, "`model` is required in request"
                            if self.model is None:
                                self.model = request_json["model"]
                                self.max_requests_per_minute = MODEL2RPM[self.model]
                                self.max_tokens_per_minute = MODEL2TPM[self.model]
                                self.available_request_capacity = self.max_requests_per_minute
                                self.available_token_capacity = self.max_tokens_per_minute

                            self.next_request = APIRequest(
                                task_id=next(self.task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, self.api_endpoint, self.model
                                ),
                                attempts_left=self.max_attempts,
                                metadata=request_json.pop("metadata", None),
                                response_to_output_func=self.response_to_output_func
                            )
                            self.status_tracker.num_tasks_started += 1
                            self.status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {self.next_request.task_id}: {self.next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            self.file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - self.last_update_time
                self.available_request_capacity = min(
                    self.available_request_capacity
                    + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute,
                )
                self.available_token_capacity = min(
                    self.available_token_capacity
                    + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
                self.last_update_time = current_time

                # if enough capacity available, call API
                if self.next_request:
                    next_request_tokens = self.next_request.token_consumption
                    if (
                        self.available_request_capacity >= 1
                        and self.available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        self.available_request_capacity -= 1
                        self.available_token_capacity -= next_request_tokens
                        self.next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            self.next_request.call_api(
                                session=session,
                                request_url=self.request_url,
                                request_header=self.request_header,
                                retry_queue=self.queue_of_requests_to_retry,
                                save_filepath=self.output_file_path,
                                status_tracker=self.status_tracker,
                                progress_bar=self.progress_bar
                            )
                        )
                        self.next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if self.status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(self.seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - self.status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < self.seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        self.seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(self.status_tracker.time_of_last_rate_limit_error + self.seconds_to_pause_after_rate_limit_error)}"
                    )

        all_done = self.status_tracker.num_tasks_total == self.status_tracker.num_tasks_succeeded
        if all_done:
            if self.post_run_func is not None:
                self.post_run_func(self.output_file_path)
            logging.info("All done!")
        else:
            assert self.status_tracker.num_tasks_failed == (self.status_tracker.num_tasks_total - self.status_tracker.num_tasks_succeeded)
            logging.info(f"{self.status_tracker.num_tasks_failed} tasks failed.")