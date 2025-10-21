from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from runner import CodeExecutor, ExecutionResult, ExecutionStage
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from contextlib import asynccontextmanager

origins = ['http://localhost:3000', 'https://prepflow.vercel.app']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExecutionRequest(BaseModel):
    language: str
    code: str
    stdin: str = ""


class ExecutionResponse(BaseModel):
    success: bool
    language: str
    stage: str
    output: str
    error: str | None = None
    compile_error: str | None = None
    execution_time: float
    return_code: int | None = None
    memory_used_kb: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Prepflow Compiler API")
    yield
    # Shutdown
    logger.info("Shutting down Prepflow Compiler API")


app = FastAPI(
    title="Prepflow Compiler API",
    description="API for executing code in multiple programming languages",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_executor() -> CodeExecutor:
    return CodeExecutor(
        timeout=3,
        memory_limit_kb=128 * 1024,
        enable_security_checks=True
    )


@app.get('/')
async def root():
    return {'message': 'Prepflow Compiler Backend Working....', 'success': True}


@app.get('/languages')
async def get_languages(executor: CodeExecutor = Depends(get_executor)):
    """Get list of supported programming languages and their configurations."""
    languages: Dict[str, Dict[str, Any]] = {}
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


@app.websocket('/ws/execute')
async def websocket_execute_code(websocket: WebSocket, executor: CodeExecutor = Depends(get_executor)):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        language = data.get('language')
        code = data.get('code')
        logger.info(f"WebSocket execution started for language: {language}")
        await executor.execute_stream(websocket, language, code)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket execution error: {str(e)}")
        try:
            await websocket.send_text(f'Error: {str(e)}')
        except WebSocketDisconnect:
            pass
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")


@app.post('/execute', response_model=ExecutionResponse)
async def execute_code(
    request: CodeExecutionRequest,
    executor: CodeExecutor = Depends(get_executor)
):
    """
    Execute code in the specified programming language.

    Available languages are dynamically determined based on system configuration.
    Use GET /languages to see available languages.
    """
    if request.language not in executor.get_supported_languages():
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language}' is not available. Supported languages are: {', '.join(executor.get_supported_languages())}"
        )

    # Special handling for TypeScript
    code = request.code
    if request.language == 'typescript' and not code.strip().startswith('// @ts-nocheck'):
        code = f"// @ts-nocheck\n{code}"

    result: ExecutionResult = executor.execute(
        language=request.language,
        code=code,
        stdin=request.stdin
    )

    # Prepare base response
    response_data = {
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

    # Clean TypeScript output if successful
    if request.language == 'typescript' and result.success and response_data['output']:
        output_lines = response_data['output'].splitlines()
        cleaned_lines = [line for line in output_lines if not line.startswith('Successfully compiled')]
        response_data['output'] = '\n'.join(cleaned_lines)

    # Handle compilation errors with formatted output
    if not result.success and result.stage == ExecutionStage.COMPILATION.value:
        error_message = result.compile_error or result.error or "Compilation failed"
        formatted_error = _format_compiler_error(error_message, request.language)

        response_data.update({
            'output': formatted_error,
            'compile_error': error_message
        })

    # Raise 400 for validation errors
    if not result.success and result.stage == ExecutionStage.VALIDATION.value:
        raise HTTPException(status_code=400, detail=result.error)

    return response_data


def _format_compiler_error(error_message: str, language: str) -> str:
    """Format compiler error messages for better readability."""
    if language not in ['typescript', 'javascript']:
        return error_message

    try:
        lines = error_message.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '): error' in line:
                # Extract location and message for TS/JS errors
                parts = line.split('): error ')
                if len(parts) == 2:
                    location_part = parts[0].rsplit('(', 1)[-1]
                    message = parts[1]
                    loc_parts = location_part.split(',')
                    if len(loc_parts) >= 2:
                        line_num, col = loc_parts[0], loc_parts[1]
                        formatted_lines.append(f"Error at line {line_num.strip()}, column {col.strip()}: {message}")
                    else:
                        formatted_lines.append(f"Error: {message}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines) if formatted_lines else error_message
    except Exception as e:
        logger.error(f"Error formatting compiler output: {str(e)}")
        return error_message