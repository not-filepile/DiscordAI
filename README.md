# DiscordAI

> **주의: 이 봇은 거친 단어를 사용할 수 있습니다.**

이 프로젝트는 텍스트 임베딩과 RAG를 이용하여 대화 내용을 기억하고 대화하는 간단한 AI 챗봇입니다.


## 기여

아직 미완성인 레포지토리로 추가적인 기능에 대한 기여를 환영합니다

## TODO

- TTS완성, 기존 TTS기능 이식
- 응답 스트리밍 구현
- rate limit을 극복하기 위한 백업 api 구현
- 실시간 임베딩 업데이트로 최신 대화 적용

## 설치 방법

1. 이 저장소를 클론합니다:

```bash
git clone https://github.com/not-filepile/DiscordAI.git
cd DiscordAI
```

2. 필요한 패키지를 설치합니다:

```bash
pip install -U discord.py google-generativeai faiss-cpu numpy tqdm openai PyNaCl
```

3. 수집한 디스코드 챗이나 각종 데이터를 불러온후 실행하기
## 기능

- RAG와 텍스트 임베딩을 사용하여 이전 디스코드 대화의 내용을 찾아 대화합니다.
- 빠른 응답 속도를 가집니다
- 간단한 커스터마이징으로 여러가지 성격의 봇을 제작할수 있습니다(system prompt수정)
