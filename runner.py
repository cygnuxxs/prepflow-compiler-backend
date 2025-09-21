import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import time
import shutil
import signal
import psutil
from dataclasses import dataclass
from enum import Enum

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
            compile_cmd=['gcc', '{file}', '-o', '{exe}', '-Wall', '-Wextra', '-std=c99', '-O2'],
            run_cmd=['{exe}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        ),
        'cpp': LanguageConfig(
            extension='.cpp',
            compile_cmd=['g++', '{file}', '-o', '{exe}', '-Wall', '-Wextra', '-std=c++17', '-O2'],
            run_cmd=['{exe}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        ),
        'java': LanguageConfig(
            extension='.java',
            compile_cmd=['javac', '{file}'],
            run_cmd=['java', '-cp', '{dir}', '-Xmx{memory_limit}m', '{main_class}'],
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
            compile_cmd=['tsc', '--lib', 'es2020,dom', '--target', 'ES2020', '--module', 'commonjs', '--strict', '{file}'],
            run_cmd=['node', '--max-old-space-size={memory_limit}', '{js_file}'],
            compiler_timeout=10,
            memory_limit_kb=128 * 1024
        )
    }

    # Security: Blacklisted patterns that might indicate malicious code
    SECURITY_BLACKLIST = [
        'import os', 'import sys', 'import subprocess', 'exec(', 'eval(',
        '__import__', 'open(', 'file(', 'input(', 'raw_input(',
        'system(', 'popen(', 'spawn', 'fork(', 'kill(',
        '#include <sys', '#include <unistd', 'system(', 'exec(',
        'Runtime.getRuntime()', 'ProcessBuilder', 'System.exit(',
        'require("fs")', 'require("child_process")', 'process.exit(',
        'std::system', 'std::exit', 'fork()', 'exec(',
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
        self.memory_limit_kb = max(64 * 1024, min(memory_limit_kb, 1024 * 1024))  # 64MB - 1GB
        self.max_output_size = max(1024, min(max_output_size, 10 * 1024 * 1024))  # 1KB - 10MB
        self.working_dir = working_dir
        self.enable_security_checks = enable_security_checks
        
        # Validate system dependencies
        self._validate_system_dependencies()

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
                logger.warning(f"Command '{cmd}' not found. Language '{lang}' will be unavailable.")

    def _check_security(self, code: str) -> None:
        """Check code for potential security issues."""
        if not self.enable_security_checks:
            return
            
        code_lower = code.lower()
        for pattern in self.SECURITY_BLACKLIST:
            if pattern.lower() in code_lower:
                raise SecurityError(f"Potentially unsafe code detected: {pattern}")

    def _format_command(self, cmd_template: List[str], **kwargs: Any) -> List[str]:
        """Format command template with variables."""
        formatted = []
        for part in cmd_template:
            if isinstance(part, str) and '{' in part:
                try:
                    formatted.append(part.format(**kwargs))
                except KeyError as e:
                    logger.warning(f"Missing template variable {e} in command: {part}")
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
        js_file = source_file.with_suffix('.js') if lang_config.extension == '.ts' else None
        
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
                logger.error(f"Compilation failed with code {result.returncode}: {error_msg}")
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
                compile_result = self._compile_code(config, tmp_path, source_file)
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
                        stdout_data, _ = process.communicate(input=stdin_data, timeout=self.timeout)
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


# Example usage
if __name__ == "__main__":
    # Example 1: Python code
    executor = CodeExecutor(timeout=10)
    
    python_code = '''
print("Hello, World!")
for i in range(3):
    print(f"Count: {i}")
'''
    
    result = executor.execute('python', python_code)
    print(f"Python result: {result.success}")
    print(f"Output: {result.output}")
    
    # Example 2: C++ code
    cpp_code = '''
#include <iostream>
using namespace std;

int main() {
    cout << "Hello from C++!" << endl;
    return 0;
}
'''
    
    result = executor.execute('cpp', cpp_code)
    print(f"C++ result: {result.success}")
    print(f"Output: {result.output}")
    
    # Show available languages
    print(f"Available languages: {executor.get_supported_languages()}")