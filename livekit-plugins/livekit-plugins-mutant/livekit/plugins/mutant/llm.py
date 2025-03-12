from __future__ import annotations

import asyncio
import datetime
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union

# Importa a biblioteca de websocket (certifique-se de que “websockets” está instalada)
import websockets

# Imports originais usados na definição de capacidades e tipos (mantidos para compatibilidade)
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.llm import (
    LLMCapabilities,
    ToolChoice,
    _create_ai_function_info,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from .log import logger
from .models import (
    ChatModels
)
# from .utils import AsyncAzureADTokenProvider

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Opcões do LLM (iguais à versão original)
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    store: bool | None = None
    metadata: dict[str, str] | None = None
    max_tokens: int | None = None

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Classe LLM – adaptada para o websocket
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class LLM(llm.LLM):
    def __init__(
        self,
        *,
        ws_uri: str,  # URL base do websocket (ex: "wss://seu_framework/ws")
        model: str | ChatModels = "gpt-4o",
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        store: bool | None = None,
        metadata: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Cria uma nova instância de LLM que se comunica via websocket com o framework.
        A conexão (única e persistente) é criada na primeira chamada e reutilizada para todas as requisições.
        """
        super().__init__(
            capabilities=LLMCapabilities(
                supports_choices_on_int=True,
                requires_persistent_functions=False,
            )
        )

        self._opts = LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            max_tokens=max_tokens,
        )
        # Armazena a URL do websocket (ex: "wss://seu_framework/ws")
        self.ws_uri = ws_uri

        # Variáveis internas para a conexão websocket
        self._ws = None                      # objeto websocket
        self._ws_incoming_queue = None       # fila para mensagens do tipo "response"
        self._ws_interrupt_queue = None      # fila para mensagens do tipo "agent_interrupt"
        self._ws_listener_task = None        # task que escuta a conexão websocket continuamente
        self._response_id = 0                # contador que serve para identificar cada mensagem enviada
        self.allow_interruptions = True      # flag para permitir interrupções

    async def _ensure_ws_connection(self):
        """
        Garante que a conexão websocket está estabelecida.
        Se não estiver, conecta utilizando uma URL que agrega um connection_id aleatório,
        inicializa as filas internas e inicia a task que ficará escutando as mensagens.
        Em seguida, envia a mensagem call_details (obrigatória, conforme o protocolo do framework).
        """
        if self._ws is None:
            connection_id = uuid.uuid4()
            ws_url = f"{self.ws_uri}/{connection_id}"
            try:
                self._ws = await websockets.connect(ws_url)
                logger.info(f"Conexão websocket persistente estabelecida: {ws_url}")
            except Exception as e:
                logger.error(f"Erro ao conectar websocket: {e}")
                raise APIConnectionError() from e

            # Inicializa as filas para mensagens recebidas
            self._ws_incoming_queue = asyncio.Queue()
            self._ws_interrupt_queue = asyncio.Queue()
            # Inicia a task que lê continuamente mensagens do websocket
            self._ws_listener_task = asyncio.create_task(self._ws_listen_loop())
            self._ws_interrupt_listener_task = asyncio.create_task(self._ws_interrupt_listener())

            # Envia o call_details uma única vez (pode ser adaptado conforme o seu projeto)
            call_details = {
                "interaction_type": "call_details",
                "call": {
                    "call_type": "web_call",
                    "call_id": f"QA_{uuid.uuid4()}",
                    "agent_id": "agent_123",
                    "call_status": "ongoing",
                    "start_timestamp": str(datetime.datetime.now()),
                    "latency": {},
                    "cost_metadata": {"llm_model": self._opts.model, "voice_provider": "custom"},
                    "call_cost": {},
                    "opt_out_sensitive_data_storage": False,
                    "access_token": "randomabc",
                },
            }
            await self._ws.send(json.dumps(call_details))

    async def _ws_listen_loop(self):
        """
        Loop que fica lendo as mensagens do websocket e encaminhando-as para
        a fila correta (ws_incoming_queue ou ws_interrupt_queue) dependendo do campo response_type.
        """
        while True:
            try:
                message = await self._ws.recv()
                try:
                    parsed = json.loads(message)
                    # logger.warning(f"Recebida mensagem JSON: {parsed}")
                except json.JSONDecodeError:
                    logger.warning("Recebida mensagem não-JSON; ignorando.")
                    continue

                if parsed.get("response_type") == "agent_interrupt":
                    if len(parsed.get("content", "").strip()) > 0 or parsed.get("content_complete"):
                        await self._ws_interrupt_queue.put(parsed)
                elif parsed.get("response_type") == "response":
                    await self._ws_incoming_queue.put(parsed)
                else:
                    pass
                # await self._ws_incoming_queue.put(parsed)
            except Exception as e:
                logger.error(f"Erro no recebimento via websocket: {e}")
                break
    
    async def _ws_interrupt_listener(self) -> None:
        """
        Task que fica escutando a fila _ws_interrupt_queue e, ao receber uma mensagem
        do tipo 'agent_interrupt', dispara o método send_proactive_message do VoicePipelineAgent.
        """
        logger.info("Iniciando task de interrupção.")
        while True:
            try:
                proactive_msg = await self._ws_interrupt_queue.get()
                if hasattr(self, "_voice_pipeline_agent"):
                    await self._voice_pipeline_agent.send_proactive_message(proactive_msg)
                else:
                    logger.error("Referência do VoicePipelineAgent não definida. Não é possível disparar mensagem proativa.")
            except Exception as e:
                logger.error(f"Erro na task de interrupção: {e}")

    def _get_next_response_id(self):
        """
        Incrementa e retorna o response_id que será utilizado no payload.
        """
        self._response_id += 1
        return self._response_id
    
    def set_voice_pipeline_agent(self, agent: "VoicePipelineAgent") -> None:
        self._voice_pipeline_agent = agent

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None = None,
    ) -> "LLMStream":
        """
        Método principal para iniciar uma “conversa”.
        Ele cria uma instância de LLMStream, que ao ser executada chamará o websocket para obter a resposta.
        """
        
        return LLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            n=n,
            temperature=temperature or self._opts.temperature,
            parallel_tool_calls=parallel_tool_calls or self._opts.parallel_tool_calls,
            tool_choice=tool_choice or self._opts.tool_choice,
        )

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Classe LLMStream – que agora usa o websocket para fazer a “stream” da resposta
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        n: int | None,
        parallel_tool_calls: bool | None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options)
        self._llm: LLM = llm
        self._chat_ctx = chat_ctx
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        # Canal de eventos em que serão “empurrados” os chunks da resposta
        # self._event_ch = asyncio.Queue()


    async def _run(self) -> None:
        """
        Implementa a lógica para esta chamada de LLM:
         1. Assegura que o websocket esteja conectado.
         2. Cria um novo response_id.
         3. Constrói um payload com a chave "interaction_type": "response_required" e envia-o.
         4. Coleta os chunks de resposta da fila ws_incoming_queue (somente os que possuem o response_id correto),
            acumulando até encontrar o chunk com "content_complete": true.
         5. Envia o chunk final para o canal de eventos (_event_ch).
        """
        # Assegura que a conexão websocket esteja ativa
        await self._llm._ensure_ws_connection()
        uid = str(uuid.uuid4())

        # Obter um response_id único para esta requisição
        response_id = self._llm._get_next_response_id()

        # Constrói o payload do request usando o chat context; adapte se necessário
        transcript = [{"role": msg.role, "content": msg.content} for msg in self._chat_ctx.messages]
        request = {
            "interaction_type": "response_required",
            "response_id": response_id,
            "timestamp": int(time.time()),
            "transcript": transcript,
        }

        try:
            await self._llm._ws.send(json.dumps(request))
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem via websocket: {e}")
            raise APIConnectionError() from e

        # Variáveis para acumular o texto de resposta e detectar flags (como no_interruption_allow)
        accumulator = ""

        # Coleta os chunks enviados via websocket que pertençam a essa chamada (usando o response_id)
        while True:
            try:
                msg = await asyncio.wait_for(self._llm._ws_incoming_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # if msg.get("response_id") != response_id:
            #     logger.debug(f"Ignorando mensagem com response_id {msg.get('response_id')} (esperado {response_id})")
            #     continue
            
            accumulator += msg.get("content", "")

            if msg.get("no_interruption_allowed", False):
                allow_interruptions = False
            else:
                allow_interruptions = True
            if allow_interruptions != self._llm.allow_interruptions:
                self._llm.allow_interruptions = allow_interruptions
                if hasattr(self._llm, "_voice_pipeline_agent"):
                    self._llm._voice_pipeline_agent.set_interruption_allowed(allow_interruptions)
                    logger.info(f"Flag de interrupção alterada para {allow_interruptions}")
                else:
                    logger.error("Referência do VoicePipelineAgent não definida. Não é possível alterar a flag de interrupção.")

            # logger.warning(f"Recebido chunk: {msg.get('content', '')}")
            
            chat_chunk = llm.ChatChunk(
                # request_id=str(response_id),
                request_id=str(uid),
                choices=[llm.Choice(delta=llm.ChoiceDelta(content=msg.get("content", ""), role="assistant"), index=0)]
            )
            self._event_ch.send_nowait(chat_chunk)

            # Se o framework enviar informações de uso (usage), estas podem ser enviadas também como evento
            usage = msg.get("usage")
            if usage:
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        request_id=str(uid),
                        usage=llm.CompletionUsage(
                            completion_tokens=usage.get("completion_tokens", 0),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        )
                    )
                )

            if msg.get("content_complete", False):
                logger.debug(f"Interrompendo loop para {'interrupt' if msg.get('response_type') == 'agent_interrupt' else 'response'}. Resposta completa: {accumulator}")
                break

    def _parse_choice(self, id: str, choice: llm.Choice) -> llm.ChatChunk | None:
        """
        Caso precise tratar chamadas de funções ou dividir os chunks, implemente esta função.
        Nesta versão simples, ela apenas empacota a mensagem recebida.
        """
        return llm.ChatChunk(
            request_id=id,
            choices=[llm.Choice(delta=llm.ChoiceDelta(content=choice.delta.content, role="assistant"), index=choice.index)]
        )

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Funções auxiliares (_build_oai_context, _strip_nones, _get_api_key) – mantidas se necessárias
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[dict[str, Any]]:
    return [{"role": msg.role, "content": msg.content} for msg in chat_ctx.messages]

def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}

def _get_api_key(env_var: str, key: str | None) -> str:
    key = key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"{env_var} is required, either as argument or set as environment variable")
    return key
