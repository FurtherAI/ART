import asyncio
import glob
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import AsyncIterator

import torch
import torchtune
from safetensors.torch import load_file
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors
from ..vllm import get_llm, get_worker, openai_server_task, run_on_workers
from .batch import Batch


@dataclass
class TorchtuneService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False

    async def start_openai_server(self, config: dev.OpenAIServerConfig | None) -> None:
        await openai_server_task(
            engine=await self.llm,
            config=dev.get_openai_server_config(
                model_name=self.model_name,
                base_model=self.get_last_checkpoint_dir() or self.base_model,
                log_file=f"{self.output_dir}/logs/vllm.log",
                config=config,
            ),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    def _log(self, msg: str) -> None:
        """Write debug log to file."""
        import datetime
        with open("service.log", "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")
            f.flush()

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        self._log("Starting train method")
        try:
            llm = await self.llm
            self._log("Got LLM engine")
        except Exception as e:
            self._log(f"Failed to get LLM: {e}")
            raise
        pids_path = f"{self.output_dir}/pids.txt"
        # reset the pids file
        with open(pids_path, "w") as f:
            f.write("")
        weights_path = "/dev/shm/weights.safetensors"
        # remove the weights file if it exists
        Path(weights_path).unlink(missing_ok=True)
        async_weight_syncing = self.torchtune_args.get("async_weight_syncing", False)
        # start putting the workers to sleep
        self._log(f"Starting sleep task for workers, async_weight_syncing={async_weight_syncing}")
        self._is_sleeping = True
        sleep_task = asyncio.create_task(
            run_on_workers(
                llm,
                sleep,
                # level=1 if llm.output_processor.has_unfinished_requests() else 2,
                level=1,
                pids_path=pids_path,
                weights_path=None if async_weight_syncing else weights_path,
                profile=verbose,
            )
        )
        # wait for the workers to write their pids twice, indicating that they are asleep
        self._log("Waiting for workers to sleep...")
        while True:
            pids = Counter(open(pids_path).read().splitlines())
            if set(pids.values()) == {2}:
                break
            await asyncio.sleep(0.25)
        self._log(f"Workers are asleep, pids: {pids}")
        # acquire the train process and queue
        self._log("Getting train process")
        try:
            train_process = await self.train_process
            self._log(f"Got train process: pid={train_process.pid}")
        except Exception as e:
            self._log(f"Failed to get train process: {e}")
            raise
        train_queue = await self.train_queue
        self._log("Got train queue, writing batch")
        # write the batch to communicate with the train process
        with open(f"{self.output_dir}/batches.jsonl", "a") as f:
            f.write(
                Batch(
                    disk_packed_tensors=disk_packed_tensors,
                    config=config,
                    dev_config=_config,
                ).model_dump_json()
                + "\n"
            )
        # consume the batch gradient step results
        self._log("Starting to consume gradient steps")
        num_gradient_steps = -1
        while num_gradient_steps != 0:
            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(train_queue.get()),
                    asyncio.create_task(train_process.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                result = task.result()
                if isinstance(result, dict):
                    result["num_gradient_steps"] = int(result["num_gradient_steps"])
                    if num_gradient_steps == -1:
                        num_gradient_steps = result["num_gradient_steps"]
                        self._log(f"Expected gradient steps: {num_gradient_steps}")
                    yield result
                else:
                    # Train process exited - get more info
                    exit_code = train_process.returncode
                    self._log(f"Train process exited with code: {exit_code}")
                    # Try to read the log file for more context
                    log_path = f"{self.output_dir}/logs/train.log"
                    try:
                        with open(log_path) as f:
                            log_content = f.read()
                            self._log(f"Last 2000 chars of train.log:\n{log_content[-2000:]}")
                    except Exception as e:
                        self._log(f"Could not read log: {e}")
                    raise RuntimeError(
                        f"Train process exited early with code {exit_code}. See {self.output_dir}/logs/train.log for details."
                    )
            num_gradient_steps -= 1
        # wait for the workers to wake up
        self._log("Training complete, waiting for workers to wake up...")
        try:
            await sleep_task
            self._log("Workers woke up successfully")
        except Exception as e:
            self._log(f"Error waking up workers: {e}")
            raise
        self._is_sleeping = False
        # update the weights after wake up if async_weight_syncing is enabled
        if async_weight_syncing:
            self._log("Starting async weight syncing")
            asyncio.create_task(self.update_worker_weights(llm, weights_path, verbose))
        else:
            self._log("Removing weights file (sync mode)")
            # remove the weights file
            Path(weights_path).unlink(missing_ok=True)

    async def update_worker_weights(
        self, llm: AsyncLLM, weights_path: str, profile: bool
    ) -> None:
        while True:
            if os.path.exists(weights_path):
                break
            else:
                time.sleep(1)
                continue
        await run_on_workers(
            llm,
            update_weights,
            weights_path=weights_path,
            profile=profile,
        )
        # remove the weights file
        Path(weights_path).unlink(missing_ok=True)

    @property
    def torchtune_args(self) -> dev.TorchtuneArgs:
        torchtune_args = self.config.get("torchtune_args")
        assert torchtune_args is not None, (
            'TorchtuneService created without config["torchtune_args"]'
        )
        return torchtune_args

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        return asyncio.create_task(
            get_llm(AsyncEngineArgs(**self.config.get("engine_args", {})))  # type: ignore
        )

    @cached_property
    def train_queue(self) -> asyncio.Task[asyncio.Queue[dict[str, float]]]:
        return asyncio.create_task(self.get_train_queue())

    @cached_property
    def train_process(self) -> asyncio.Task[asyncio.subprocess.Process]:
        return asyncio.create_task(self.get_train_process())

    async def get_train_process(self) -> asyncio.subprocess.Process:
        # Migrate existing checkpoints to new structure if needed
        from ..local.checkpoints import migrate_checkpoints_to_new_structure

        migrate_checkpoints_to_new_structure(self.output_dir)

        Path(f"{self.output_dir}/batches.jsonl").unlink(missing_ok=True)
        checkpoint_dir = await self.get_checkpoint_dir()
        torchtune_args = self.torchtune_args

        # Get the list of safetensor files
        safetensor_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        checkpoint_files = [os.path.basename(f) for f in safetensor_files]
        checkpoint_files_str = "[" + ", ".join(f'"{f}"' for f in checkpoint_files) + "]"

        def model_dir(model: str) -> str:
            for prefix in [
                "llama3_1",
                "llama3_2_vision",
                "llama3_2",
                "llama3_3",
                "qwen2_5",
            ]:
                if model.startswith(prefix):
                    return prefix
            return model.split("_")[0]

        program_and_args = [
            "python",  # Use Python interpreter
            f"{os.path.dirname(torchtune.__file__)}/_cli/tune.py",
            "run",
            "--nproc-per-node",
            str(torch.cuda.device_count()),
            "art.torchtune.recipe.FullFinetuneRecipeDistributed",
            "--config",
            f"{os.path.dirname(__file__)}/config.yaml",
            f"model._component_=torchtune.models.{model_dir(torchtune_args['model'])}.{torchtune_args['model']}",
            f"checkpointer.checkpoint_dir={checkpoint_dir}",
            f"checkpointer.checkpoint_files={checkpoint_files_str}",
            f"checkpointer.model_type={torchtune_args['model_type']}",
            f"tensor_parallel_dim={torchtune_args.get('tensor_parallel_dim', 1)}",
            f"context_parallel_dim={torchtune_args.get('context_parallel_dim', 1)}",
            f"output_dir={self.output_dir}",
            "clip_grad_norm=0.1",
            "metric_logger._component_=torchtune.training.metric_logging.StdoutLogger",
            "metric_logger.log_dir=null",
            f"enable_activation_offloading={torchtune_args.get('enable_activation_offloading', False)}",
        ]
        return await asyncio.subprocess.create_subprocess_exec(
            *program_and_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def get_train_queue(self) -> asyncio.Queue[dict[str, float]]:
        process = await self.train_process
        queue = asyncio.Queue()

        async def read(reader: asyncio.StreamReader) -> None:
            async for line in reader:
                line_str = line.decode("utf-8")
                with open(f"{self.output_dir}/logs/train.log", "a") as f:
                    f.write(line_str)
                line_str = line_str.strip()
                if line_str.startswith("Step ") and " | " in line_str:
                    parts = line_str.split(" | ", 1)
                    metrics: dict[str, float] = {}
                    if len(parts) > 1:
                        for metric in parts[1].split():
                            if ":" in metric:
                                name, value = metric.split(":", 1)
                                try:
                                    metrics[name] = float(value)
                                except ValueError:
                                    # Skip non-numeric values to match the return type
                                    pass
                    await queue.put(metrics)

        assert process.stdout and process.stderr
        asyncio.create_task(read(process.stdout))
        asyncio.create_task(read(process.stderr))
        return queue

    async def get_checkpoint_dir(self) -> str:
        # Use the last of any existing checkpoints to resume training
        if last_checkpoint_dir := self.get_last_checkpoint_dir():
            return last_checkpoint_dir
        # Check if self.base_model is a directory
        if os.path.isdir(self.base_model):
            return self.base_model
        # Otherwise, assume it's a HuggingFace model id and download it
        process = await asyncio.subprocess.create_subprocess_exec(
            "hf",
            "download",
            "--repo-type=model",
            self.base_model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return stdout.decode("utf-8").splitlines()[-1].strip()

    def get_last_checkpoint_dir(self) -> str | None:
        from ..local.checkpoints import get_last_checkpoint_dir

        return get_last_checkpoint_dir(self.output_dir)


def _worker_log(msg: str) -> None:
    """Write debug log from worker process."""
    import datetime
    with open("service.log", "a") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] [worker-{os.getpid()}] {msg}\n")
        f.flush()


def sleep(
    *, level: int, pids_path: str, weights_path: str | None, profile: bool
) -> None:
    """
    Put the worker to sleep until the new model weights are loaded.

    Args:
        level: The sleep level: 1 to offload the kv cache, 2 to discard the kv cache.
        pids_path: The path to the file that contains the PIDs of the workers.
        weights_path: The path to the weights file.
        profile: Whether to profile
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.worker.gpu_worker import logger

    _worker_log(f"sleep() called: level={level}, weights_path={weights_path}")
    with open(pids_path, "a") as f:
        f.write(f"{os.getpid()}\n")
    worker = get_worker()
    allocator = CuMemAllocator.get_instance()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        setattr(allocator, "_override_tags", {"weights", "kv_cache"})
        _worker_log("Calling worker.sleep()")
        with worker.time("sleep"):
            worker.sleep(level)
        _worker_log("worker.sleep() returned")
        with open(pids_path, "a") as f:
            f.write(f"{os.getpid()}\n")
        weights = None
        _worker_log(f"Waiting for weights or pids_path removal...")
        while True:
            if weights_path:
                # wait for the weights file to be created
                try:
                    _worker_log(f"Trying to load weights from {weights_path}")
                    with worker.time("load_file"):
                        weights = load_file(weights_path)
                    _worker_log("Weights loaded successfully")
                    break
                except FileNotFoundError:
                    time.sleep(1)
                    continue
            elif os.path.exists(pids_path):
                time.sleep(1)
                continue
            else:
                # no pids file indicates we can wake up
                _worker_log("pids_path removed, waking up")
                break
        _worker_log("Calling worker.wake_up()")
        with worker.time("wake_up"):
            worker.wake_up()
        _worker_log("worker.wake_up() returned")
        if weights is None:
            _worker_log("No weights to load, returning")
            return
        _worker_log("Loading weights into model")
        with worker.time("load_weights"):
            worker.model_runner.model.load_weights(weights.items())  # type: ignore
        _worker_log("Weights loaded into model successfully")
    except Exception as e:
        _worker_log(f"ERROR in sleep(): {e}")
        raise
    finally:
        logger.setLevel(logging.INFO)
        delattr(allocator, "_override_tags")


def update_weights(weights_path: str, profile: bool) -> None:
    from vllm.v1.worker.gpu_worker import logger

    worker = get_worker()
    try:
        if not (profile and worker.rank == 0):
            logger.setLevel(logging.CRITICAL)
        with worker.time("load_file"):
            weights = load_file(weights_path)
        with worker.time("load_weights"):
            worker.model_runner.model.load_weights(weights.items())  # type: ignore
    finally:
        logger.setLevel(logging.INFO)
