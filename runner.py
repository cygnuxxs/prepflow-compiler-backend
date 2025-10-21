import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from fastapi import WebSocket
import logging
import asyncio
import time
import shutil
import signal
import psutil
from dataclasses import dataclass
from enum import Enum
from starlette.websockets import WebSocketDisconnect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStage(Enum):
    """Enumeration for execution stages."""
    COMPILATION = "compilation"
    EXECUTION = "execution"
    VALIDATION = "validation"


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""
    extension: str
    run_cmd: List[str]
    compile_cmd: Optional[List[str]] = None
    main_class: Optional[str] = None
    compiler_timeout: Optional[int] = None
    memory_limit_kb: int = 128 * 1024
    requires_main_class: bool = False


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    language: str
    stage: str
    return_code: Optional[int] = None
    output: str = ""
    error: Optional[str] = None
    compile_error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_kb: Optional[int] = None


class SecurityError(Exception):
    """Raised when security constraints are violated."""
    pass


class CodeExecutor:
    """Secure code execution sandbox for multiple languages."""

    # Language configuration using dataclass for better type safety
    LANGUAGES: Dict[str, LanguageConfig] = {
        'python': LanguageConfig(
            extension='.py',
            run_cmd=['python3', '{file}'],
            memory_limit_kb=128 * 1024
        ),
        'c': LanguageConfig(
            extension='.c',
            compile_cmd=['gcc', '{file}', '-o', '{exe}',
                         '-Wall', '-Wextra', '-std=c99', '-O2'],
            run_cmd=['{exe}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        ),
        'cpp': LanguageConfig(
            extension='.cpp',
            compile_cmd=['g++', '{file}', '-o', '{exe}',
                         '-Wall', '-Wextra', '-std=c++17', '-O2'],
            run_cmd=['{exe}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        ),
        'java': LanguageConfig(
            extension='.java',
            compile_cmd=['javac', '{file}'],
            run_cmd=['java', '-cp', '{dir}',
                     '-Xmx{memory_limit}m', '{main_class}'],
            main_class='Main',
            compiler_timeout=10,
            memory_limit_kb=192 * 1024,
            requires_main_class=True
        ),
        'javascript': LanguageConfig(
            extension='.js',
            run_cmd=['node', '--max-old-space-size={memory_limit}', '{file}'],
            memory_limit_kb=128 * 1024
        ),
        'typescript': LanguageConfig(
            extension='.ts',
            compile_cmd=['tsc', '--lib', 'es2020,dom', '--target',
                         'ES2020', '--module', 'commonjs', '--strict', '{file}'],
            run_cmd=[
                'node', '--max-old-space-size={memory_limit}', '{js_file}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        )
    }

    # Security: Blacklisted patterns that might indicate malicious code
    SECURITY_BLACKLIST = [
        'import os', 'import sys', 'import subprocess', 'exec(',
        '__import__', 'open(', 'file(',
        'system(', 'popen(', 'spawn', 'fork(', 'kill(',
        '#include <sys', '#include <unistd',
        'Runtime.getRuntime()', 'ProcessBuilder', 'System.exit(',
        'require("fs")', 'require("child_process")', 'process.exit(',
        'std::system', 'std::exit', 'exit('
    ]

    def __init__(self,
                 timeout: int = 5,
                 memory_limit_kb: int = 256 * 1024,
                 max_output_size: int = 1024 * 1024,
                 working_dir: Optional[str] = None,
                 enable_security_checks: bool = True):
        """
        Initialize CodeExecutor.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_kb: Memory limit in KB
            max_output_size: Maximum output size in bytes
            working_dir: Working directory for temporary files
            enable_security_checks: Whether to perform security checks
        """
        self.timeout = max(1, min(timeout, 30))  # Clamp between 1-30 seconds
        self.memory_limit_kb = max(
            64 * 1024, min(memory_limit_kb, 1024 * 1024))  # 64MB - 1GB
        self.max_output_size = max(
            1024, min(max_output_size, 10 * 1024 * 1024))  # 1KB - 10MB
        self.working_dir = working_dir
        self.enable_security_checks = enable_security_checks

        # Validate system dependencies
        self._validate_system_dependencies()

    async def execute_stream(self, websocket: WebSocket, language: str, code: str):
        """Stream execution output and handle live input over WebSocket."""
        # Validate inputs
        if not code.strip():
            await websocket.send_text("No code provided")
            return

        if language not in self.LANGUAGES:
            await websocket.send_text(f"Unsupported language: {language}. Available: {list(self.LANGUAGES.keys())}")
            return

        if language not in self.available_languages:
            await websocket.send_text(f"Language '{language}' is not available on this system")
            return

        config = self.LANGUAGES[language]

        try:
            # Security check
            self._check_security(code)

            # Java-specific validation
            if language == 'java' and not self._validate_java_class(code):
                await websocket.send_text("Java code must contain 'public class Main' with 'public static void main' method")
                return

        except SecurityError as e:
            await websocket.send_text(f"Security check failed: {str(e)}")
            return

        # Create temporary directory
        with tempfile.TemporaryDirectory(dir=self.working_dir) as tmpdir:
            tmp_path = Path(tmpdir)

            try:
                # Write source code
                source_file = tmp_path / self._get_safe_filename(language)
                source_file.write_text(code, encoding='utf-8')
                logger.info(f"Created source file: {source_file}")

                # Compile if necessary
                compile_result = self._compile_code(
                    config, tmp_path, source_file)
                if compile_result["error"]:
                    await websocket.send_text(f"Compilation failed:\n{compile_result['error']}")
                    return

                executable = compile_result["executable"]

                # Prepare execution command
                memory_limit_mb = self.memory_limit_kb // 1024
                format_args = {
                    'file': str(source_file),
                    'exe': executable,
                    'js_file': executable,
                    'dir': str(tmp_path),
                    'main_class': config.main_class,
                    'memory_limit': memory_limit_mb
                }

                run_cmd = self._format_command(config.run_cmd, **format_args)
                logger.info(f"Executing with command: {' '.join(run_cmd)}")

                start_time = time.time()
                # Track last activity (stdout/read or stdin/write) to implement inactivity-based timeout
                last_activity = time.monotonic()

                # Start subprocess with its own process group/session so we can signal it
                create_kwargs = {}
                if os.name != 'nt':
                    create_kwargs['start_new_session'] = True
                else:
                    try:
                        create_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                    except AttributeError:
                        create_kwargs['creationflags'] = 0

                process = await asyncio.create_subprocess_exec(
                    *run_cmd,
                    cwd=str(tmp_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    # Merge stderr to stdout to match synchronous behavior
                    stderr=asyncio.subprocess.STDOUT,
                    **create_kwargs
                )

                # Define pipe reader (now only stdout since merged)
                async def read_pipe(pipe, ws):
                    while True:
                        try:
                            chunk = await pipe.read(1024)
                            if len(chunk) == 0:
                                break
                            text = chunk.decode('utf-8', errors='replace')
                            # Apply output size limit per chunk for safety
                            text = self._limit_output_size(text)
                            # Update last activity on output
                            nonlocal last_activity
                            last_activity = time.monotonic()
                            await ws.send_text(text)
                        except WebSocketDisconnect:
                            break
                        except Exception as e:
                            logger.error(f"Error reading pipe: {e}")
                            break

                # Define input handler with polling to check process status
                async def handle_input(ws, stdin, proc):
                    """Handle WebSocket input and forward to process stdin."""
                    input_queue = asyncio.Queue()

                    async def interrupt_process():
                        """Send SIGINT (Ctrl+C) to the running process with graceful fallback."""
                        try:
                            if os.name != 'nt':
                                try:
                                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                                except OSError:
                                    # If process group signal fails, try direct
                                    os.kill(proc.pid, signal.SIGINT)
                            else:
                                # On Windows, prefer CTRL_BREAK_EVENT when in a new process group
                                try:
                                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                                except Exception:
                                    proc.terminate()
                        except Exception as e:
                            logger.error(f"Error sending interrupt: {e}")
                        
                        # Give a short grace period; if still alive, escalate
                        try:
                            for _ in range(10):  # ~1s total
                                if proc.returncode is not None:
                                    return
                                await asyncio.sleep(0.1)
                            # Escalate to SIGTERM / terminate
                            if os.name != 'nt':
                                try:
                                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                except OSError:
                                    try:
                                        os.kill(proc.pid, signal.SIGTERM)
                                    except Exception:
                                        proc.terminate()
                            else:
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"Error escalating termination: {e}")
                    
                    async def receive_input():
                        """Continuously receive input from WebSocket."""
                        try:
                            while True:
                                data = await ws.receive_text()
                                await input_queue.put(data)
                        except WebSocketDisconnect:
                            await input_queue.put(None)  # Signal disconnect
                        except Exception as e:
                            logger.error(f"Receive error: {e}")
                            await input_queue.put(None)
                    
                    # Start input receiver task
                    receiver_task = asyncio.create_task(receive_input())
                    
                    try:
                        while proc.returncode is None:
                            try:
                                # Wait for input with a short timeout to check process status
                                data = await asyncio.wait_for(input_queue.get(), timeout=0.1)
                                
                                if data is None:  # Disconnected
                                    break
                                
                                # Normalize data for control checks (don't strip control chars)
                                trimmed = data.strip().lower() if data is not None else ""

                                # Check for exit command
                                if trimmed == "exit()":
                                    if os.name != 'nt':
                                        try:
                                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                        except OSError:
                                            proc.terminate()
                                    else:
                                        proc.terminate()
                                    await ws.send_text("\nProcess terminated by user.\n")
                                    break

                                # Check for Ctrl+C / SIGINT requests
                                # Accept any of: actual ETX (\x03), "^C", "ctrl+c", "SIGINT"
                                is_ctrl_c = (
                                    data == "\x03" or
                                    trimmed in ("^c", "ctrl+c", "sigint")
                                )
                                if is_ctrl_c:
                                    await ws.send_text("\n^C\n")
                                    await interrupt_process()
                                    break
                                
                                # Write input to process stdin
                                # Ensure newline so languages waiting on line-buffered input (e.g. Python input()) proceed
                                if not data.endswith("\n"):
                                    data = data + "\n"
                                stdin.write(data.encode('utf-8'))
                                await stdin.drain()
                                # Update last activity on input
                                nonlocal last_activity
                                last_activity = time.monotonic()
                                
                            except asyncio.TimeoutError:
                                # No input received, continue checking if process is alive
                                continue
                                
                    finally:
                        receiver_task.cancel()
                        try:
                            await receiver_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            stdin.close()
                        except:
                            pass

                # Define memory monitor
                async def memory_monitor(proc, ws, limit_kb):
                    while proc.returncode is None:
                        await asyncio.sleep(0.5)
                        try:
                            p = psutil.Process(proc.pid)
                            mem = p.memory_info().rss // 1024
                            if mem > limit_kb:
                                if os.name != 'nt':
                                    try:
                                        os.killpg(os.getpgid(
                                            proc.pid), signal.SIGTERM)
                                    except OSError:
                                        proc.terminate()
                                else:
                                    proc.terminate()
                                await ws.send_text(f"\nMemory limit exceeded ({mem}KB > {limit_kb}KB)\n")
                                break
                        except psutil.NoSuchProcess:
                            break
                        except Exception as e:
                            logger.error(f"Mem monitor error: {e}")

                # Create tasks
                stdout_task = asyncio.create_task(
                    read_pipe(process.stdout, websocket))
                input_task = asyncio.create_task(
                    handle_input(websocket, process.stdin, process))
                mem_task = asyncio.create_task(memory_monitor(
                    process, websocket, self.memory_limit_kb))

                # Inactivity-based timeout watchdog (prevents killing while user is typing)
                async def inactivity_watchdog():
                    # Use a more lenient threshold for interactive sessions
                    threshold = max(self.timeout, 30)
                    try:
                        while process.returncode is None:
                            await asyncio.sleep(0.25)
                            if time.monotonic() - last_activity > threshold:
                                if os.name != 'nt':
                                    try:
                                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                                    except OSError:
                                        process.terminate()
                                else:
                                    process.terminate()
                                await websocket.send_text(f"\nExecution timed out after {threshold} seconds of inactivity\n")
                                break
                    except Exception as e:
                        logger.error(f"Watchdog error: {e}")

                timeout_task = asyncio.create_task(inactivity_watchdog())

                # Run IO tasks
                try:
                    await asyncio.gather(stdout_task, input_task, return_exceptions=True)
                finally:
                    # Cleanup tasks
                    for task in [stdout_task, input_task]:
                        if not task.done():
                            task.cancel()
                    try:
                        await process.wait()
                    except:
                        pass

                # Cancel monitors
                timeout_task.cancel()
                try:
                    await timeout_task
                except asyncio.CancelledError:
                    pass
                mem_task.cancel()
                try:
                    await mem_task
                except asyncio.CancelledError:
                    pass

                # Send final info
                exec_time = time.time() - start_time
                message = f"""
Process exited with code {process.returncode}
Execution time: {round(exec_time, 3)}"""
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Unexpected error in execute_stream: {str(e)}")
                await websocket.send_text(f"Unexpected error: {str(e)}")

    def _validate_system_dependencies(self) -> None:
        """Check if required system dependencies are available."""
        required_commands = {
            'python3': 'python',
            'gcc': 'c',
            'g++': 'cpp',
            'javac': 'java',
            'java': 'java',
            'node': 'javascript',
            'tsc': 'typescript'
        }

        self.available_languages = set()
        for cmd, lang in required_commands.items():
            if shutil.which(cmd):
                self.available_languages.add(lang)
            else:
                logger.warning(
                    f"Command '{cmd}' not found. Language '{lang}' will be unavailable.")

    def _check_security(self, code: str) -> None:
        """Check code for potential security issues."""
        if not self.enable_security_checks:
            return

        code_lower = code.lower()
        for pattern in self.SECURITY_BLACKLIST:
            if pattern.lower() in code_lower:
                raise SecurityError(
                    f"Potentially unsafe code detected: {pattern}")

    def _format_command(self, cmd_template: List[str], **kwargs: Any) -> List[str]:
        """Format command template with variables."""
        formatted = []
        for part in cmd_template:
            if isinstance(part, str) and '{' in part:
                try:
                    formatted.append(part.format(**kwargs))
                except KeyError as e:
                    logger.warning(
                        f"Missing template variable {e} in command: {part}")
                    formatted.append(part)
            else:
                formatted.append(str(part))
        return formatted

    def _validate_java_class(self, code: str) -> bool:
        """Validate that Java code has proper Main class structure."""
        lines = code.strip().split('\n')
        has_main_class = False
        has_main_method = False

        for line in lines:
            line = line.strip()
            if line.startswith('public class Main'):
                has_main_class = True
            elif 'public static void main' in line:
                has_main_method = True

        return has_main_class and has_main_method

    def _compile_code(self, lang_config: LanguageConfig, tmpdir: Path, source_file: Path) -> Dict[str, Any]:
        """Compile code if needed. Returns dict with executable path and error info."""
        if not lang_config.compile_cmd:
            return {"executable": str(source_file), "error": None}

        # Determine executable path and additional files
        exe = tmpdir / 'program'
        js_file = source_file.with_suffix(
            '.js') if lang_config.extension == '.ts' else None

        # Special handling for different languages
        if lang_config.extension == '.java':
            executable = str(tmpdir)
        elif lang_config.extension == '.ts':
            executable = str(js_file)
        else:
            executable = str(exe)

        # Format compilation command
        format_args = {
            'file': str(source_file),
            'exe': str(exe),
            'js_file': str(js_file) if js_file else '',
            'dir': str(tmpdir)
        }

        cmd = self._format_command(lang_config.compile_cmd, **format_args)

        timeout = lang_config.compiler_timeout or self.timeout
        try:
            logger.info(f"Compiling with command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                logger.error(
                    f"Compilation failed with code {result.returncode}: {error_msg}")
                return {"executable": None, "error": error_msg}

            # Verify executable was created
            if lang_config.extension in ['.c', '.cpp', '.rs', '.go'] and not Path(exe).exists():
                return {"executable": None, "error": "Executable was not created"}
            elif lang_config.extension == '.ts' and not (js_file and js_file.exists()):
                return {"executable": None, "error": "JavaScript file was not created from TypeScript"}

            return {"executable": executable, "error": None}

        except subprocess.TimeoutExpired:
            error_msg = f"Compilation timed out after {timeout} seconds"
            logger.error(error_msg)
            return {"executable": None, "error": error_msg}
        except Exception as e:
            error_msg = f"Compilation error: {str(e)}"
            logger.error(error_msg)
            return {"executable": None, "error": error_msg}

    def _limit_output_size(self, output: str) -> str:
        """Limit output size to prevent memory issues."""
        if len(output) > self.max_output_size:
            truncated = output[:self.max_output_size]
            return truncated + f"\n[Output truncated at {self.max_output_size} characters]"
        return output

    def _get_safe_filename(self, language: str) -> str:
        """Get safe filename for the given language."""
        config = self.LANGUAGES.get(language)
        if not config:
            raise ValueError(f"Unsupported language: {language}")

        # Use Main for Java, otherwise use program
        base_name = "Main" if language == "java" else "program"
        return f"{base_name}{config.extension}"

    def _monitor_process_memory(self, process: subprocess.Popen) -> Optional[int]:
        """Monitor process memory usage."""
        try:
            ps_process = psutil.Process(process.pid)
            memory_info = ps_process.memory_info()
            return memory_info.rss // 1024  # Convert to KB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def execute(self, language: str, code: str, stdin: str = "") -> ExecutionResult:
        """
        Execute code in the specified language.

        Args:
            language: Programming language
            code: Source code to execute
            stdin: Input data for the program

        Returns:
            ExecutionResult object with execution details
        """
        # Validate inputs
        if not code.strip():
            return ExecutionResult(
                success=False,
                language=language,
                stage=ExecutionStage.VALIDATION.value,
                error="No code provided"
            )

        if language not in self.LANGUAGES:
            return ExecutionResult(
                success=False,
                language=language,
                stage=ExecutionStage.VALIDATION.value,
                error=f"Unsupported language: {language}. Available: {list(self.LANGUAGES.keys())}"
            )

        if language not in self.available_languages:
            return ExecutionResult(
                success=False,
                language=language,
                stage=ExecutionStage.VALIDATION.value,
                error=f"Language '{language}' is not available on this system"
            )

        config = self.LANGUAGES[language]

        try:
            # Security check
            self._check_security(code)

            # Java-specific validation
            if language == 'java' and not self._validate_java_class(code):
                return ExecutionResult(
                    success=False,
                    language=language,
                    stage=ExecutionStage.VALIDATION.value,
                    error="Java code must contain 'public class Main' with 'public static void main' method"
                )

        except SecurityError as e:
            return ExecutionResult(
                success=False,
                language=language,
                stage=ExecutionStage.VALIDATION.value,
                error=f"Security check failed: {str(e)}"
            )

        # Create temporary directory and execute
        with tempfile.TemporaryDirectory(dir=self.working_dir) as tmpdir:
            tmp_path = Path(tmpdir)

            try:
                # Write source code
                source_file = tmp_path / self._get_safe_filename(language)
                source_file.write_text(code, encoding='utf-8')
                logger.info(f"Created source file: {source_file}")

                # Compile if necessary
                compile_result = self._compile_code(
                    config, tmp_path, source_file)
                if compile_result["error"]:
                    return ExecutionResult(
                        success=False,
                        language=language,
                        stage=ExecutionStage.COMPILATION.value,
                        compile_error=compile_result["error"],
                        error="Compilation failed"
                    )

                executable = compile_result["executable"]

                # Prepare execution command
                memory_limit_mb = self.memory_limit_kb // 1024
                format_args = {
                    'file': str(source_file),
                    'exe': executable,
                    'js_file': executable if language == 'typescript' else str(tmp_path / 'output.js'),
                    'dir': str(tmp_path),
                    'main_class': config.main_class,
                    'memory_limit': memory_limit_mb
                }

                run_cmd = self._format_command(config.run_cmd, **format_args)
                logger.info(f"Executing with command: {' '.join(run_cmd)}")

                # Execute the program
                stdin_data = stdin.encode('utf-8') if stdin else b''
                start_time = time.time()

                try:
                    process = subprocess.Popen(
                        run_cmd,
                        cwd=str(tmp_path),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid if os.name != 'nt' else None
                    )

                    # Monitor execution
                    try:
                        stdout_data, _ = process.communicate(
                            input=stdin_data, timeout=self.timeout)
                        return_code = process.returncode
                    except subprocess.TimeoutExpired:
                        # Kill process group to ensure cleanup
                        if os.name != 'nt':
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        else:
                            process.terminate()
                        process.wait()

                        return ExecutionResult(
                            success=False,
                            language=language,
                            stage=ExecutionStage.EXECUTION.value,
                            error=f"Execution timed out after {self.timeout} seconds",
                            execution_time=self.timeout
                        )

                    exec_time = time.time() - start_time
                    output = stdout_data.decode('utf-8', errors='replace')
                    output = self._limit_output_size(output)

                    return ExecutionResult(
                        success=return_code == 0,
                        language=language,
                        stage=ExecutionStage.EXECUTION.value,
                        return_code=return_code,
                        output=output,
                        execution_time=round(exec_time, 3),
                        error=None if return_code == 0 else f"Process exited with code {return_code}"
                    )

                except Exception as e:
                    logger.error(f"Execution error: {str(e)}")
                    return ExecutionResult(
                        success=False,
                        language=language,
                        stage=ExecutionStage.EXECUTION.value,
                        error=f"Execution failed: {str(e)}"
                    )

            except Exception as e:
                logger.error(f"Unexpected error in execute: {str(e)}")
                return ExecutionResult(
                    success=False,
                    language=language,
                    stage=ExecutionStage.EXECUTION.value,
                    error=f"Unexpected error: {str(e)}"
                )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported and available languages."""
        return sorted(self.available_languages)

    def get_language_info(self, language: str) -> Optional[LanguageConfig]:
        """Get configuration info for a specific language."""
        return self.LANGUAGES.get(language)


# Convenience wrapper function
def run_code(language: str, code: str, stdin: str = "", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to execute code.

    Args:
        language: Programming language
        code: Source code
        stdin: Input data
        **kwargs: Additional arguments for CodeExecutor

    Returns:
        Dictionary representation of ExecutionResult
    """
    executor = CodeExecutor(**kwargs)
    result = executor.execute(language, code, stdin)

    # Convert ExecutionResult to dictionary for backward compatibility
    return {
        'success': result.success,
        'language': result.language,
        'stage': result.stage,
        'return_code': result.return_code,
        'output': result.output,
        'error': result.error,
        'compile_error': result.compile_error,
        'execution_time': result.execution_time,
        'memory_used_kb': result.memory_used_kb
    }
