import os
import json
import numpy as np
import faiss
import asyncio
import hashlib
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class AIService:
    def __init__(
            self,
            embedding_function,
            cache_dir: str = "embedding_cache",
            max_context_size: int = 5 # 이거 올리면 더 많은 대화를 가져올 수 있기는 한데 더 느려질 수 있음
    ):
        """
        :param embedding_function: 임베딩 생성 함수
        :param cache_dir: 임베딩 캐시를 저장할 디렉토리
        :param max_context_size: RAG 검색 시 가져올 최대 컨텍스트 수
        """
        self.embedding_function = embedding_function
        self.max_context_size = max_context_size

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 저장
        self.conversation_chunks = []
        self.index = None

        # 파일 경로
        self.embeddings_file = self.cache_dir / "embeddings.pkl"
        self.index_file = self.cache_dir / "faiss_index.bin"
        self.chunks_file = self.cache_dir / "conversation_chunks.json"

        self.is_ready = False

    def _get_cache_key(self, text: str) -> str:
        """텍스트에 대한 캐시 키 생성 (MD5 해시)"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        :param text: 임베딩할 텍스트
        :param use_cache: 캐시 사용 여부
        :return: 임베딩 벡터
        """
        if not text.strip():
            return [0.0] * 768

        # 캐시설정
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"캐시 로드 오류 ({cache_key}): {e}")

        try:
            # 비동기 함수로 임베딩 생성
            embedding = await self.embedding_function(text)

            # 캐시 저장
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)

            return embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            # 오류 발생 시 기본 차원의 0 벡터 반환
            return [0.0] * 768

    async def load_discord_data(self, file_paths: List[str], chunk_size: int = 3, overlap: int = 1,
                                use_cache: bool = True):
        """
        :param file_paths: 데이터 파일 경로 목록
        :param chunk_size: 대화 청크 크기
        :param overlap: 겹치는 청크 수
        :param use_cache: 이전에 생성된 임베딩 캐시 및 인덱스 사용 여부
        """
        # sonnet이 한거
        if use_cache and await self._try_load_cached_data():
            print("캐시에서 임베딩 및 인덱스 데이터를 불러왔습니다.")
            self.is_ready = True
            return

        all_messages = []

        # 파일에서 메시지 로드
        for file_path in file_paths:
            try:
                if file_path.endswith('.json'):
                    # JSON 파일 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 형식에 따라 처리
                        if isinstance(data, list) and len(data) > 0 and 'content' in data[0]:
                            for msg in data:
                                if msg.get('content', '').strip():
                                    all_messages.append(msg['content'])

                elif file_path.endswith('.txt'):
                    # 텍스트 파일 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 대화 구분 (빈 줄 기준)
                        messages = re.split(r'\n\s*\n', content)
                        for msg in messages:
                            if msg.strip():
                                all_messages.append(msg.strip())

                print(f"파일 로드 완료: {file_path} ({len(all_messages)}개 메시지)")

            except Exception as e:
                print(f"파일 로드 오류 ({file_path}): {e}")

        # 대화 청킹
        self.conversation_chunks = []

        for i in range(0, len(all_messages) - chunk_size + 1, chunk_size - overlap):
            chunk = all_messages[i:i + chunk_size]
            chunk_text = "\n\n".join(chunk)
            self.conversation_chunks.append({
                "text": chunk_text,
                "messages": chunk
            })

        print(f"대화 청크 생성 완료: {len(self.conversation_chunks)}개")

        # 대화 청크 저장
        await self._save_conversation_chunks()

        # 벡터 인덱스 생성
        await self._create_vector_index(use_cache=use_cache)

        self.is_ready = True

    async def _save_conversation_chunks(self):
        """대화 청크 저장"""
        try:
            # JSON으로 저장 가능한 형태로 변환
            serializable_chunks = []
            for chunk in self.conversation_chunks:
                serializable_chunks.append({
                    "text": chunk["text"],
                    "messages": chunk["messages"]
                })

            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

            print(f"대화 청크 저장 완료: {self.chunks_file}")
        except Exception as e:
            print(f"대화 청크 저장 오류: {e}")

    async def _create_vector_index(self, use_cache: bool = True):
        """
        대화 청크에 대한 FAISS 벡터 인덱스 생성

        :param use_cache: 개별 임베딩에 캐시 사용 여부
        """
        print("벡터 인덱스 생성 중...")

        if not self.conversation_chunks:
            print("인덱싱할 대화 청크가 없습니다.")
            return

        # 각 청크를 임베딩
        embeddings = []

        for chunk in tqdm(self.conversation_chunks, desc="청크 임베딩 중"):
            try:
                embedding = await self.embed_text(chunk["text"], use_cache=use_cache)
                embeddings.append(embedding)
            except Exception as e:
                print(f"청크 임베딩 오류: {e}")
                # 오류 발생 시 0 벡터로 대체
                embedding_dim = 768  # 기본 임베딩 차원
                if embeddings and len(embeddings[0]) > 0:
                    embedding_dim = len(embeddings[0])
                embeddings.append([0.0] * embedding_dim)

        # FAISS 인덱스 생성
        if embeddings:
            embedding_dim = len(embeddings[0])
            self.index = faiss.IndexFlatL2(embedding_dim)
            embeddings_np = np.array(embeddings).astype('float32')
            self.index.add(embeddings_np)
            print(f"벡터 인덱스 생성 완료: {len(embeddings)}개 임베딩")

            # 임베딩 및 인덱스 저장
            await self._save_embeddings_and_index(embeddings)
        else:
            print("임베딩 생성 실패")

    async def _save_embeddings_and_index(self, embeddings: List[List[float]]):
        """임베딩과 FAISS 인덱스를 파일로 저장"""
        try:
            # 임베딩 저장
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"임베딩 저장 완료: {self.embeddings_file}")

            # FAISS 인덱스 저장
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
                print(f"FAISS 인덱스 저장 완료: {self.index_file}")
        except Exception as e:
            print(f"임베딩 및 인덱스 저장 오류: {e}")

    async def _try_load_cached_data(self) -> bool:
        """
        캐시에서 대화 청크, 임베딩, 인덱스 로드 시도

        :return: 로드 성공 여부
        """
        try:
            # 1. 대화 청크 로드
            if not self.chunks_file.exists():
                return False

            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.conversation_chunks = json.load(f)

            # 2. FAISS 인덱스 로드
            if not self.index_file.exists():
                return False

            # 비동기 환경에서 FAISS 로드는 블로킹 작업이므로 실행자에서 실행
            loop = asyncio.get_event_loop()
            self.index = await loop.run_in_executor(
                None,
                lambda: faiss.read_index(str(self.index_file))
            )

            print(f"캐시에서 {len(self.conversation_chunks)}개 대화 청크와 FAISS 인덱스({self.index.ntotal}개 벡터) 로드 완료")
            return True

        except Exception as e:
            print(f"캐시 데이터 로드 오류: {e}")
            return False

    async def get_relevant_context(self, query: str, k: int = None):
        """
        질의와 관련된 가장 연관성 높은 대화 컨텍스트 검색

        :param query: 사용자 질의
        :param k: 검색할 관련 컨텍스트 수 (None이면 self.max_context_size 사용)
        :return: 관련 컨텍스트 텍스트
        """
        if not self.index or not self.conversation_chunks:
            return ""

        if k is None:
            k = self.max_context_size

        try:
            # 질의 임베딩
            query_embedding = await self.embed_text(query)
            query_embedding_np = np.array([query_embedding]).astype('float32')

            # 유사 컨텍스트 검색 (비동기 환경에서 FAISS 검색은 블로킹 작업)
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.index.search(query_embedding_np, k)
            )
            distances, indices = search_results

            # 관련 컨텍스트 추출
            contexts = []
            for idx in indices[0]:
                if 0 <= idx < len(self.conversation_chunks):
                    contexts.append(self.conversation_chunks[idx]["text"])

            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            print(f"관련 컨텍스트 검색 오류: {e}")
            return ""

    async def clear_cache(self, confirm: bool = False) -> bool:
        """
        임베딩 캐시 삭제

        :param confirm: 확인 없이 삭제 진행 여부
        :return: 성공 여부
        """
        if not confirm:
            return False

        try:
            # 모든 .pkl 파일 삭제
            for cache_file in self.cache_dir.glob("*.pkl"):
                os.remove(cache_file)

            # FAISS 인덱스 파일 삭제
            if self.index_file.exists():
                os.remove(self.index_file)

            # 청크 파일 삭제
            if self.chunks_file.exists():
                os.remove(self.chunks_file)

            print(f"캐시 디렉토리 {self.cache_dir}의 모든 캐시가 삭제되었습니다.")
            return True
        except Exception as e:
            print(f"캐시 삭제 중 오류 발생: {e}")
            return False