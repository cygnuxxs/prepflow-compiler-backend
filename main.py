from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from runner import CodeExecutor, ExecutionResult
import logging
from fastapi.middleware.cors import CORSMiddleware

origins = ['http://localhost:3000', 'https://prepflow.vercel.app']

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Prepflow Compiler API",
              description="API for executing code in multiple programming languages",
              version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"], )
# Initialize the code executor with security checks enabled
executor = CodeExecutor(
    timeout=3,
    memory_limit_kb=128 * 1024,
    enable_security_checks=True
)


class CodeExecutionRequest(BaseModel):
    language: str
    code: str
    stdin: str = ""  # Default empty string instead of Optional


@app.get('/')
async def root():
    return {'message': 'Prepflow Compiler Backend Working....', 'success': True}


@app.get('/languages')
async def get_languages():
    """Get list of supported programming languages and their configurations."""
    languages = {}
    for lang in executor.get_supported_languages():
        config = executor.get_language_info(lang)
        if config:
            languages[lang] = {
                'extension': config.extension,
                'requires_compilation': bool(config.compile_cmd),
                'requires_main_class': config.requires_main_class,
                'memory_limit_kb': config.memory_limit_kb,
            }
    return {'languages': languages}


@app.post('/execute')
async def execute_code(request: CodeExecutionRequest):
    """
    Execute code in the specified programming language.

    Available languages are dynamically determined based on system configuration.
    Use GET /languages to see available languages.

    Returns:
        The execution result including:
        - success: boolean indicating if the execution was successful
        - language: the programming language used
        - stage: the stage where execution ended (validation/compilation/execution)
        - output: program output or compilation output
        - error: error message if execution failed
        - compile_error: detailed compiler output if compilation failed
        - execution_time: time taken to execute in seconds
        - return_code: program return code (0 for success)
        - memory_used_kb: peak memory usage in KB (if available)
    """
    try:
        if request.language not in executor.get_supported_languages():
            raise HTTPException(
                status_code=400,
                detail=f"Language '{request.language}' is not available. Supported languages are: {', '.join(executor.get_supported_languages())}"
            )

        # Special handling for TypeScript to ensure proper compilation
        if request.language == 'typescript':
            if not request.code.strip().startswith('// @ts-nocheck'):
                request.code = f"// @ts-nocheck\n{request.code}"

        result: ExecutionResult = executor.execute(
            language=request.language,
            code=request.code,
            stdin=request.stdin
        )

        # Convert ExecutionResult to dictionary for API response
        response = {
            'success': result.success,
            'language': result.language,
            'stage': result.stage,
            'output': result.output,
            'error': result.error,
            'compile_error': result.compile_error,
            'execution_time': result.execution_time,
            'return_code': result.return_code,
            'memory_used_kb': result.memory_used_kb
        }

        # Clean up typescript output if needed
        if request.language == 'typescript' and response['output']:
            output_lines = response['output'].splitlines()
            cleaned_lines = [line for line in output_lines
                             if not line.startswith('Successfully compiled')]
            response['output'] = '\n'.join(cleaned_lines)

        # For compilation errors, format and return the error details
        if not result.success and result.stage == 'compilation':
            error_message = result.compile_error or result.error or "Compilation failed"
            formatted_error = error_message

            # Process TypeScript/JavaScript errors
            if request.language in ['typescript', 'javascript']:
                try:
                    # Extract just the filename from the full path
                    lines = error_message.split('\n')
                    formatted_lines = []

                    for line in lines:
                        if '): error' in line:
                            # Extract error details
                            parts = line.split('): error ')
                            if len(parts) == 2:
                                location = parts[0].split('(')[-1]
                                message = parts[1]
                                # Convert location like "2,20" to "line 2, column 20"
                                loc_parts = location.split(',')
                                if len(loc_parts) == 2:
                                    line_num, col = loc_parts
                                    formatted_lines.append(
                                        f"Error at line {line_num}, column {col}: {message}")
                                else:
                                    formatted_lines.append(f"Error: {message}")
                        elif line.strip():  # Keep any non-empty lines that aren't error lines
                            formatted_lines.append(line)

                    if formatted_lines:
                        formatted_error = '\n'.join(formatted_lines)

                except Exception as e:
                    logger.error(f"Error formatting compiler output: {str(e)}")
                    # Fall back to original error message if formatting fails
                    formatted_error = error_message

            return {
                'success': False,
                'language': result.language,
                'stage': result.stage,
                'error': result.error,
                'compile_error': error_message,
                'output': formatted_error
            }

        # For validation errors, return 400 status
        if not result.success and result.stage == 'validation':
            raise HTTPException(
                status_code=400,
                detail=result.error
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
