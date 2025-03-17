import asyncio
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GeminiClient:
    """Gemini API 호출"""

    def __init__(
            self,
            api_key: str,
            model_name: str = "models/gemini-2.0-flash-lite",
            embedding_model: str = "models/text-embedding-004",
            temperature: float = 0.7,
            max_output_tokens: int = 128,
            concurrent_requests: int = 5  # 동시 요청 처리 수, 많아지면 레이트 리밋 걸릴수도 있음
    ):
        """
        :param api_key: Gemini API 키
        :param model_name: 사용할 Gemini 모델명
        :param embedding_model: 사용할 임베딩 모델명
        :param temperature: 생성 온도
        :param max_output_tokens: 최대 출력 토큰 수
        :param concurrent_requests: 동시에 처리할 수 있는 최대 요청 수
        """
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model

        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
        }

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.request_semaphore = asyncio.Semaphore(concurrent_requests)

        self.model = self._create_model()

    def _create_model(self):
        """Gemini 모델 인스턴스 생성"""
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

    def set_api_key(self, api_key: str):
        """
        :param api_key: 새 API 키
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = self._create_model()

    async def embed_content(self, text: str) -> List[float]:
        """
        :param text: 임베딩할 텍스트
        :return: 임베딩 벡터
        """
        async with self.request_semaphore:
            loop = asyncio.get_event_loop()
            try:
                embedding_result = await loop.run_in_executor(
                    None,
                    lambda: genai.embed_content(
                        model=self.embedding_model,
                        content=text,
                        task_type="RETRIEVAL_QUERY"
                    )
                )
                return embedding_result["embedding"]
            except Exception as e:
                raise Exception(f"임베딩 생성 실패: {e}")

    async def generate_response(
            self,
            messages: List[Dict[str, str]],
            system_prompt: str = None,
            context: str = None,
            user_info: Dict[str, str] = None,
    ) -> str:
        """
        :param messages: 대화 메시지 목록
        :param system_prompt: 시스템 프롬프트
        :param context: RAG 컨텍스트 (있는 경우)
        :param user_info: 사용자 정보
        :return: 생성된 응답
        """
        async with self.request_semaphore:
            try:
                chat = self.model.start_chat(history=[])

                prompt_parts = []

                if system_prompt:
                    prompt_parts.append(system_prompt)

                if user_info:
                    user_info_text = "\n".join([f"{k}: {v}" for k, v in user_info.items()])
                    prompt_parts.append(f"사용자 정보:\n{user_info_text}")

                if context:
                    prompt_parts.append(f"""
다음은 이전 대화에서 현재 질문과 관련된 맥락입니다:

{context}

위 맥락을 참고하여 답변해주세요.
""")

                if prompt_parts:
                    system_message = "\n\n".join(prompt_parts)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: chat.send_message(system_message)
                    )

                for msg in messages:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: chat.send_message(msg.get("content", ""))
                    )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: chat.send_message(messages[-1].get("content", ""))
                )

                return response.text

            except Exception as e:
                print(f"응답 생성 실패: {e}")
                return "나중에 다시 시도해주세요."

    async def generate_response_stream(
            self,
            messages: List[Dict[str, str]],
            system_prompt: str = None,
            context: str = None,
            user_info: Dict[str, str] = None,
    ):
        """
        :param messages: 대화 메시지 목록
        :param system_prompt: 시스템 프롬프트
        :param context: RAG 컨텍스트 (있는 경우)
        :param user_info: 사용자 정보
        :yield: 생성된 응답 청크
        """
        async with self.request_semaphore:
            try:
                # 채팅 인스턴스 생성
                chat = self.model.start_chat(history=[])

                # 프롬프트 구성
                prompt_parts = []

                # 시스템 프롬프트 추가
                if system_prompt:
                    prompt_parts.append(system_prompt)

                # 사용자 정보 추가
                if user_info:
                    user_info_text = "\n".join([f"{k}: {v}" for k, v in user_info.items()])
                    prompt_parts.append(f"사용자 정보:\n{user_info_text}")

                # 컨텍스트 추가
                if context:
                    prompt_parts.append(f"""
다음은 이전 대화에서 현재 질문과 관련된 맥락입니다:

{context}

위 맥락을 참고하여 답변해주세요.
""")

                # 시스템 프롬프트 전송
                if prompt_parts:
                    system_message = "\n\n".join(prompt_parts)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: chat.send_message(system_message)
                    )

                # 이전 메시지 전송
                for msg in messages[:-1]:  # 마지막 메시지 제외
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: chat.send_message(msg.get("content", ""))
                    )

                # 응답 스트리밍 준비
                last_message = messages[-1].get("content", "")

                # 백그라운드 태스크로 스트리밍 실행
                loop = asyncio.get_event_loop()

                # 스트리밍 응답 가져오기
                response_stream = await loop.run_in_executor(
                    None,
                    lambda: chat.send_message(last_message, stream=True)
                )

                # 스트림에서 청크 가져오기
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text

            except Exception as e:
                # 오류 발생 시 메시지 반환
                print(f"스트리밍 응답 생성 실패: {e}")
                yield "죄송합니다. 지금은 응답을 생성할 수 없습니다. 나중에 다시 시도해주세요."