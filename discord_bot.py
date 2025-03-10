import os
import re
import json
import asyncio
import signal
import sys
import discord
from discord.ext import commands
from pathlib import Path
import subprocess

from ai_service import AIService
from gemini_client import GeminiClient


# tts, 구현중
class BetterFFmpegPCMAudio(discord.FFmpegOpusAudio):
    def __init__(self, source, **kwargs):
        self._process = None
        self._process_pid = None
        super().__init__(source, **kwargs)

    def _spawn_process(self, args, **subprocess_kwargs):
        self._process = super()._spawn_process(args, **subprocess_kwargs)
        self._process_pid = self._process.pid
        return self._process

    def cleanup(self):# tts용
        try:
            if self._process:
                super().cleanup()

                try:
                    if self._process_pid:
                        if sys.platform == 'win32':
                            subprocess.call(['taskkill', '/F', '/T', '/PID', str(self._process_pid)],
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                        else:
                            try:
                                os.kill(self._process_pid, signal.SIGKILL)
                            except:
                                pass
                except:
                    pass
        except Exception as e:
            print(f"프로세스 종료 중 오류: {e}")


class GeminiDiscordBot:
    def __init__(
            self,
            discord_token: str,
            gemini_api_key: str,
            model_name: str = "models/gemini-2.0-flash-lite",
            embedding_model: str = "models/text-embedding-004",
            system_prompt: str = None,
            backup_api_keys: list = None,
            cache_dir: str = "embedding_cache",
            max_context_size: int = 5,
            enable_tts: bool = False,
            tts_voice: str = None
    ):
        """
        :param discord_token: Discord 봇 토큰
        :param gemini_api_key: Gemini API 키
        :param model_name: 사용할 Gemini 모델명
        :param embedding_model: 사용할 임베딩 모델명
        :param system_prompt: 시스템 프롬프트 (없으면 기본값 사용)
        :param backup_api_keys: 대체 API 키 목록
        :param cache_dir: 임베딩 캐시를 저장할 디렉토리
        :param max_context_size: RAG 검색 시 가져올 최대 컨텍스트 수
        :param enable_tts: TTS 기능 활성화 여부
        :param tts_voice: TTS 음성 (None이면 기본값)
        """
        self.discord_token = discord_token

        if system_prompt is None:
            self.system_prompt = """
            [시스템 프롬프트]
            Discord 채널의 대화 맥락과 분위기를 이해하고, 그에 맞게 대응해 주세요.
            사용자의 질문에 최대한 자연스럽고 맥락에 맞게 답변하세요.
            이름을 언급할때 [이름]야,아 등 적절한 조사를 붙여라

            [기본적인 유저 정보]
            사용자의 닉네임: [부를떄 사용할 사용자 이름]

            [내용]
            """
        else:
            self.system_prompt = system_prompt

        self.gemini_client = GeminiClient(
            api_key=gemini_api_key,
            model_name=model_name,
            embedding_model=embedding_model,
            backup_api_keys=backup_api_keys or []
        )

        self.ai_service = AIService(
            embedding_function=self.gemini_client.embed_content,
            cache_dir=cache_dir,
            max_context_size=max_context_size
        )

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.voice_states = True

        self.bot = commands.Bot(command_prefix="!", intents=intents)

        self.channel_chat_history = {}

        self.is_processing = {}

        self.enable_tts = enable_tts
        self.tts_voice = tts_voice
        self.voice_connections = {}

        self.setup_event_handlers()

    def setup_event_handlers(self):
        """Discord 이벤트 핸들러 설정"""

        @self.bot.event
        async def on_ready():
            print(f"봇 준비 완료 {self.bot.user}")

            try:
                synced = await self.bot.tree.sync()
                print(f"{len(synced)}개 명령어 동기화 완료")
            except Exception as e:
                print(f"명령어 동기화 실패: {e}")

            try:
                # 데이터 파일 목록
                discord_data_files = [
                    "임베딩에 사용할 데이터 파일 경로",
                ]

                await self.ai_service.load_discord_data(
                    file_paths=discord_data_files,
                    chunk_size=3,
                    overlap=1,
                    use_cache=True
                )
                print("AI 서비스 준비 완료!")
            except Exception as e:
                print(f"데이터 로드 오류: {e}")

        @self.bot.event
        async def on_message(message):
            await self.bot.process_commands(message)

            if message.author == self.bot.user:
                return

            # 봇이 멘션된 경우에만 응답
            if (self.bot.user.mentioned_in(message) or
                re.search(r'<@1048855298154180628>', message.content)) and not message.mention_everyone:
                await self.process_mention(message)

    async def process_mention(self, message):
        content = re.sub(r'<@!?1048855298154180628>', '', message.content).strip()
        content = re.sub(rf'<@!?{self.bot.user.id}>', '', content).strip()

        mentions = re.findall(r'<@!?(\d+)>', content)
        for user_id in mentions:
            try:
                member = message.guild.get_member(int(user_id))
                if member:
                    content = content.replace(f'<@{user_id}>', f'@{member.display_name}')
                    content = content.replace(f'<@!{user_id}>', f'@{member.display_name}')
            except:
                pass

        # print(f"처리할 메시지: {content}")

        if not content:  # 멘션만 있는 경우
            await message.channel.send("꺼져 ㅋㅋ")
            return

        if not self.ai_service.is_ready:
            await message.channel.send("잠시 후 다시 시도해주세요.")
            return

        # 사용자 닉네임 가져오기
        user_name = message.author.display_name
        # 메시지 로그 저장(학습용)
        with open("message.txt", "a") as f:
            f.write(user_name + ": " + content + "\n")
        previous_content = ""
        if message.reference and message.reference.resolved:
            previous_content = message.reference.resolved.content

            # 멘션 닉네임으로 변환
            mentions = re.findall(r'<@!?(\d+)>', previous_content)
            for user_id in mentions:
                try:
                    member = message.guild.get_member(int(user_id))
                    if member:
                        previous_content = previous_content.replace(f'<@{user_id}>', f'@{member.display_name}')
                        previous_content = previous_content.replace(f'<@!{user_id}>', f'@{member.display_name}')
                except:
                    pass

            content = f"[이전내용]\n{previous_content}\n\n[대답]\n{content}"

        content_with_username = f"[사용자: {user_name}]\n{content}"

        # 타이핑 상태
        async with message.channel.typing():
            channel_id = str(message.channel.id)

            response = await self.chat(
                channel_id=channel_id,
                user_input=content_with_username,
                user_name=user_name,
                use_rag=True
            )
            print(f"프롬프트: {content_with_username}")
            # 응답 전송
            try:
                if len(response) > 500:  # 긴 응답은 여러 메시지로 나누어 전송
                    chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
                    await message.reply(chunks[0])
                else:
                    if "죄송합니다" not in response:
                        # 특정 단어/표현 필터링
                        filtered_response = response.replace("니 애미는 니 같은 놈 낳고 얼마나 좆같을까.", "")
                        filtered_response = filtered_response.replace("니 애미는 널 낳고 얼마나 좆같을까.", "")
                        filtered_response = filtered_response.replace(". ", " ")
                        filtered_response = filtered_response.replace(", ", " ")
                        filtered_response = filtered_response.replace("@everyone", "@애브리원")
                        filtered_response = filtered_response.replace("@here", "@히얼")
                        # filtered_response = filtered_response.replace("<@", "")
                        filtered_response = filtered_response.replace("낄", "ㅋ")
                        filtered_response = filtered_response.replace("니 애미는 이런 놈 낳고 얼마나 좆같을까.", "")

                        await message.reply("# " + filtered_response)

                        if self.enable_tts and message.guild.voice_client:
                            await self.play_tts(message.guild.voice_client, filtered_response)
                    else:
                        await message.reply("다시 시도해주세요.")
            except Exception as e:
                print(f"응답 전송 오류: {e}")
                await message.reply("응답 전송 중 오류가 발생했습니다.")

    async def chat(
            self,
            channel_id: str,
            user_input: str,
            use_rag: bool = True,
            k: int = None,
            user_name: str = None,
            return_thinking: bool = False
    ):
        """
        :param channel_id: 채널 ID
        :param user_input: 사용자 입력
        :param use_rag: RAG 사용 여부
        :param k: 검색할 관련 컨텍스트 수
        :param user_name: 사용자 이름
        :param return_thinking: 생각 과정 반환 여부
        :return: 생성된 응답 또는 응답+생각 과정 딕셔너리
        """
        if channel_id in self.is_processing and self.is_processing[channel_id]:
            return "꺼져 ㅋ"

        self.is_processing[channel_id] = True

        try:
            # 채널별 대화 기록 초기화 (없는 경우)
            if channel_id not in self.channel_chat_history:
                self.channel_chat_history[channel_id] = []

            # 대화 히스토리 관리
            if len(self.channel_chat_history[channel_id]) > 20:
                self.channel_chat_history[channel_id] = self.channel_chat_history[channel_id][-20:]

            # 관련 컨텍스트 검색 (RAG 활성화된 경우)
            context = ""
            thinking = ""

            if use_rag and self.ai_service.is_ready:
                thinking += "🔍 관련 컨텍스트 검색 중...\n"
                context = await self.ai_service.get_relevant_context(user_input, k=k)

                if context:
                    thinking += f"✅ {k or self.ai_service.max_context_size}개의 관련 컨텍스트를 찾았습니다.\n"
                else:
                    thinking += "관련 컨텍스트를 찾지 못했습니다.\n"

            messages = []
            for msg in self.channel_chat_history[channel_id]:
                messages.append({"content": msg})

            messages.append({"content": user_input})

            user_info = None
            if user_name:
                user_info = {"이름": user_name}

            thinking += "응답 생성 중...\n"
            response = await self.gemini_client.generate_response(
                messages=messages,
                system_prompt=self.system_prompt,
                context=context,
                user_info=user_info
            )

            self.channel_chat_history[channel_id].append(user_input)
            self.channel_chat_history[channel_id].append(response)

            if return_thinking:
                return {"response": response, "thinking": thinking}
            return response

        except Exception as e:
            print(f"채팅 오류: {e}")
            return "나중에 다시 시도해주세요."

        finally:
            self.is_processing[channel_id] = False

    async def clear_channel_history(self, channel_id: str) -> bool:
        try:
            if channel_id in self.channel_chat_history:
                self.channel_chat_history[channel_id] = []
            return True
        except Exception as e:
            print(f"채널 기록 초기화 오류: {e}")
            return False

    async def play_tts(self, voice_client, text):
        if not voice_client or not voice_client.is_connected():
            return False

        try:
            tts_file = Path("")

            # TODO TTS API 호출 구현
            if tts_file.exists():
                if voice_client.is_playing():
                    voice_client.stop()

                # 재생
                audio_source = discord.FFmpegPCMAudio(str(tts_file))
                voice_client.play(audio_source)
                return True
            return False
        except Exception as e:
            print(f"TTS 재생 오류: {e}")
            return False

    def run(self):
        self.bot.run(self.discord_token)


if __name__ == "__main__":
    discord_token = os.getenv("DISCORD_TOKEN",
                              "DISCORD_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY")

    backup_api_keys = ["BACKUP_API_KEY_1", "BACKUP_API_KEY_2"]

    gemini_bot = GeminiDiscordBot(
        discord_token=discord_token,
        gemini_api_key=gemini_api_key,
        model_name="models/gemini-2.0-flash-lite",
        embedding_model="models/text-embedding-004",
        backup_api_keys=backup_api_keys,
        cache_dir="discord_bot_cache",
        max_context_size=5,
        enable_tts=False  # 이거 True해도 맥 아니면 안될수도 있음, FFmpeg 설치해야함
    )

    gemini_bot.run()